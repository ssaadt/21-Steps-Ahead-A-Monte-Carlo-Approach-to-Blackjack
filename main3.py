import random
from collections import defaultdict, namedtuple
import time

NUM_DECKS = 1
RESHUFFLE_CUT = 5
TRAIN_EPISODES = 1_000_000
MIN_BET = 10
MAX_BET_FRAC = 0.05

class Shoe:
    def __init__(self, num_decks=NUM_DECKS, reshuffle_cut=RESHUFFLE_CUT):
        self.num_decks = num_decks
        self.reshuffle_cut = reshuffle_cut
        self.cards = []
        self.running_count = 0
        self.needs_reshuffle = False
        self._build_shoe()

    def _build_shoe(self):
        self.cards = []
        for _ in range(self.num_decks):
            for rank in range(1, 14):
                card_value = 10 if rank >= 10 else rank
                for _ in range(4):
                    self.cards.append(card_value)
        random.shuffle(self.cards)
        self.running_count = 0
        self.needs_reshuffle = False

    def decks_remaining(self):
        return len(self.cards) / 52.0

    def true_count(self):
        decks = max(self.decks_remaining(), 1e-9)
        return self.running_count / decks

    def draw(self):
        if not self.cards:
            self._build_shoe()

        card = self.cards.pop()

        if 2 <= card <= 6:
            self.running_count += 1
        elif card == 1 or card == 10:
            self.running_count -= 1

        if len(self.cards) < self.reshuffle_cut:
            self.needs_reshuffle = True

        return card

    def start_new_hand(self):
        if self.needs_reshuffle:
            self._build_shoe()
            self.needs_reshuffle = False

SHOE = Shoe(num_decks=NUM_DECKS, reshuffle_cut=RESHUFFLE_CUT)

State = namedtuple("State", ["player_sum", "usable_ace", "dealer_upcard", "tc_bucket"])
ACTIONS = ["hit", "stand"]

def draw_card(shoe=None):
    if shoe is None:
        shoe = SHOE
    return shoe.draw()

def draw_hand(shoe=None):
    return [draw_card(shoe), draw_card(shoe)]

def usable_ace(hand):
    return 1 in hand and sum(hand) + 10 <= 21

def sum_hand(hand):
    s = sum(hand)
    if 1 in hand and s + 10 <= 21:
        return s + 10
    return s

def is_bust(hand):
    return sum_hand(hand) > 21

def dealer_play(hand, shoe=None):
    while sum_hand(hand) < 17:
        hand.append(draw_card(shoe))
    return hand

def score(player_hand, dealer_hand):
    p = sum_hand(player_hand)
    d = sum_hand(dealer_hand)
    if p > 21:
        return -1
    if d > 21:
        return 1
    if p > d:
        return 1
    if p < d:
        return -1
    return 0

def tc_to_bucket(tc):
    if tc <= -2:
        return 0
    if -2 < tc <= 0:
        return 1
    if 0 < tc <= 1:
        return 2
    if 1 < tc <= 2:
        return 3
    if 2 < tc <= 3:
        return 4
    if 3 < tc <= 5:
        return 5
    return 6

def state_from_hands(player_hand, dealer_upcard, shoe=None):
    if shoe is None:
        shoe = SHOE
    tc = shoe.true_count()
    bucket = tc_to_bucket(tc)
    return State(sum_hand(player_hand), usable_ace(player_hand), dealer_upcard, bucket)

def greedy_action(Q, state):
    q_hit = Q[state]["hit"]
    q_stand = Q[state]["stand"]
    if q_hit > q_stand:
        return "hit"
    if q_stand > q_hit:
        return "stand"
    return random.choice(ACTIONS)

def epsilon_greedy_action(Q, state, epsilon):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return greedy_action(Q, state)

def generate_episode(policy_fn, exploring_starts=False, shoe=None):
    if shoe is None:
        shoe = SHOE

    shoe.start_new_hand()
    player = draw_hand(shoe)
    dealer = draw_hand(shoe)
    dealer_up = dealer[0]

    episode = []
    first_decision = True

    while True:
        s = state_from_hands(player, dealer_up, shoe)
        if first_decision and exploring_starts:
            a = random.choice(ACTIONS)
        else:
            a = policy_fn(s)

        if a == "hit":
            player.append(draw_card(shoe))
            if is_bust(player):
                episode.append((s, a, -1))
                return episode
            episode.append((s, a, 0))
        else:
            dealer = dealer_play(dealer, shoe)
            episode.append((s, a, score(player, dealer)))
            return episode

        first_decision = False

def train_mc_onpolicy(num_episodes=TRAIN_EPISODES, epsilon=0.1, exploring_starts=False, print_every=50_000, shoe=None):
    if shoe is None:
        shoe = SHOE

    def q_init():
        return {"hit": 0.1, "stand": 0.1}

    Q = defaultdict(lambda: q_init())
    N = defaultdict(lambda: {"hit": 0, "stand": 0})

    def policy_fn(state):
        return epsilon_greedy_action(Q, state, epsilon)

    for ep in range(1, num_episodes + 1):
        episode = generate_episode(policy_fn, exploring_starts, shoe)
        final_reward = episode[-1][2]
        visited = set()
        for (s, a, _) in episode:
            if (s, a) in visited:
                continue
            visited.add((s, a))
            N[s][a] += 1
            Q[s][a] += (final_reward - Q[s][a]) / N[s][a]

        if ep % print_every == 0:
            print(f"Trained {ep}/{num_episodes} episodes... (running_count={shoe.running_count:.2f}, decks_left={shoe.decks_remaining():.2f})")

    def learned_policy(state):
        if state not in Q:
            return "hit" if state.player_sum < 17 else "stand"
        return greedy_action(Q, state)

    return Q, learned_policy

def naive_baseline_policy(state):
    return "hit" if state.player_sum < 17 else "stand"

def random_policy(state):
    return random.choice(ACTIONS)

def play_episode_with_policy(policy_fn, shoe=None):
    if shoe is None:
        shoe = SHOE

    shoe.start_new_hand()
    player = draw_hand(shoe)
    dealer = draw_hand(shoe)
    dealer_up = dealer[0]

    while True:
        s = state_from_hands(player, dealer_up, shoe)
        a = policy_fn(s)
        if a == "hit":
            player.append(draw_card(shoe))
            if is_bust(player):
                return -1
        else:
            dealer = dealer_play(dealer, shoe)
            return score(player, dealer)

def evaluate_policy(policy_fn, num_games=10000, num_decks=NUM_DECKS, reshuffle_cut=RESHUFFLE_CUT):
    eval_shoe = Shoe(num_decks=num_decks, reshuffle_cut=reshuffle_cut)
    wins = ties = losses = total_reward = 0
    for _ in range(num_games):
        r = play_episode_with_policy(policy_fn, eval_shoe)
        total_reward += r
        if r == 1:
            wins += 1
        elif r == 0:
            ties += 1
        else:
            losses += 1
    return {
        "games": num_games,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": wins / num_games,
        "average_reward": total_reward / num_games
    }

def format_card(card):
    return 'A' if card == 1 else str(card)

def format_hand(hand):
    return '[' + ', '.join(format_card(c) for c in hand) + ']'

def q_to_win_prob(q):
    return max(0.0, min(1.0, (q + 1) / 2))

def ai_recommendation(Q, state, bankroll):
    if state not in Q:
        return "hit", 0.5, MIN_BET

    q_hit = Q[state]["hit"]
    q_stand = Q[state]["stand"]

    action = "hit" if q_hit >= q_stand else "stand"
    win_prob = q_to_win_prob(max(q_hit, q_stand))

    edge = win_prob - 0.5
    raw_bet = bankroll * MAX_BET_FRAC * max(edge * 2, 0)
    bet = max(MIN_BET, int(raw_bet))
    bet = min(bet, bankroll)

    return action, win_prob, bet


def play_interactive_game(Q, learned_policy):
    shoe = Shoe()
    bankroll = 1000
    base_bet = 10
    hands = wins = losses = pushes = 0

    while bankroll > 0:
        shoe.start_new_hand()
        player = draw_hand(shoe)
        dealer = draw_hand(shoe)
        dealer_up = dealer[0]
        hands += 1

        state = state_from_hands(player, dealer_up, shoe)
        q_hit = Q[state]["hit"]
        q_stand = Q[state]["stand"]
        ai_action = learned_policy(state)
        q_best = max(q_hit, q_stand)
        win_prob = q_to_win_prob(q_best)
        
        print(f"\nBankroll: ${bankroll}")
        print(f"Dealer shows: {format_card(dealer_up)}")
        print(f"Your hand: {format_hand(player)} ({sum_hand(player)})")

        ai_action, win_prob, bet_rec = ai_recommendation(Q, state, bankroll)
        print(f"AI recommends: {ai_action.upper()} | Win Prob: {win_prob*100:.1f}% | Bet: ${bet_rec}")

        while True:
            bet_input = input(f"Enter your bet (min ${MIN_BET}): ").strip()
            if bet_input.isdigit():
                bet = int(bet_input)
                if MIN_BET <= bet <= bankroll:
                    break
            print("Invalid bet.")

        bankroll -= bet

        
        while True:
            state = state_from_hands(player, dealer_up, shoe)
            ai_action, win_prob, _ = ai_recommendation(Q, state, bankroll)
            print(f"AI recommends: {ai_action.upper()} | Win Prob: {win_prob*100:.1f}%")
            choice = input("Hit or Stand? [H/S]: ").lower()
            if choice.startswith("h"):
                player.append(draw_card(shoe))
                print(f"Your hand: {format_hand(player)} ({sum_hand(player)})")
                if is_bust(player):
                    print("BUST")
                    losses += 1
                    break
            else:
                dealer = dealer_play(dealer, shoe)
                result = score(player, dealer)
                if result == 1:
                    bankroll += bet * 2
                    wins += 1
                    print("You WIN!")
                elif result == 0:
                    bankroll += bet
                    pushes += 1
                    print("PUSH (TIE)")
                else:
                    losses += 1
                    print("You LOSE!")
                break

        print(f"Stats: {wins}W {losses}L {pushes}P")

        
        if input("Play again? [Y/n]: ").lower().startswith("n"):
            break

if __name__ == "__main__":
    EPSILON = 0.1
    EXPLORING_STARTS = True
    EVAL_GAMES = 50_000

    # Ask user what they want to do
    print("="*60)
    print("BLACKJACK MONTE CARLO AI")
    print("="*60)
    print("\nWhat would you like to do?")
    print("1. Train AI and see statistics")
    print("2. Train AI, then play interactive game")

    choice = input("\nEnter choice (1/2): ").strip()

    if choice == "1":
        # Original behavior - train and evaluate
        print("\nTraining Monte Carlo agent with true-count buckets (6-deck shoe, realistic reshuffle)...")
        t0 = time.time()
        Q, learned_policy = train_mc_onpolicy(num_episodes=TRAIN_EPISODES,
                                              epsilon=EPSILON,
                                              exploring_starts=EXPLORING_STARTS,
                                              print_every=50_000,
                                              shoe=SHOE)
        t1 = time.time()
        print(f"Training done in {t1 - t0:.1f}s")

        print("\nEvaluating learned policy on a fresh shoe...")
        results_learned = evaluate_policy(learned_policy, num_games=EVAL_GAMES, num_decks=NUM_DECKS, reshuffle_cut=RESHUFFLE_CUT)
        print("Learned policy:", results_learned)

        print("\nEvaluating naive baseline (hit < 17) on a fresh shoe...")
        results_baseline = evaluate_policy(naive_baseline_policy, num_games=EVAL_GAMES, num_decks=NUM_DECKS, reshuffle_cut=RESHUFFLE_CUT)
        print("Baseline policy:", results_baseline)

        print("\nEvaluating random policy on a fresh shoe...")
        results_random = evaluate_policy(random_policy, num_games=EVAL_GAMES, num_decks=NUM_DECKS, reshuffle_cut=RESHUFFLE_CUT)
        print("Random policy:", results_random)

        print("\nSample policy (player_sum, usable_ace, dealer_upcard, tc_bucket) -> action (learned)")
        samples = [
            State(12, False, 2, tc_to_bucket(-1.5)),
            State(12, False, 10, tc_to_bucket(2.2)),
            State(16, False, 10, tc_to_bucket(0.0)),
            State(18, False, 9, tc_to_bucket(4.5)),
            State(13, True, 6, tc_to_bucket(1.1)),
            State(20, False, 10, tc_to_bucket(-3.0))
        ]
        for s in samples:
            print(s, "->", learned_policy(s))

    elif choice == "2":
        # Full training then play
        print("\nTraining Monte Carlo agent with true-count buckets (6-deck shoe, realistic reshuffle)...")
        t0 = time.time()
        Q, learned_policy = train_mc_onpolicy(num_episodes=TRAIN_EPISODES,
                                              epsilon=EPSILON,
                                              exploring_starts=EXPLORING_STARTS,
                                              print_every=50_000,
                                              shoe=SHOE)
        t1 = time.time()
        print(f"Training done in {t1 - t0:.1f}s")

        # Start interactive game
        play_interactive_game(Q, learned_policy)

    else:
        print("Invalid choice. Exiting.")
