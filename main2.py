import random
from collections import defaultdict, namedtuple
import time

NUM_DECKS = 6
RESHUFFLE_CUT = 5
TRAIN_EPISODES = 1_000_000   


class Shoe:
    """
    Finite multi-deck blackjack shoe using card values:
      Ace = 1, 2-9 = pip, 10 covers 10,J,Q,K
    Maintains Hi-Lo running_count and supports realistic reshuffle (between hands).
    """
    def __init__(self, num_decks=NUM_DECKS, reshuffle_cut=RESHUFFLE_CUT):
        self.num_decks = num_decks
        self.reshuffle_cut = reshuffle_cut 
        self.cards = []
        self.running_count = 0
        self.needs_reshuffle = False
        self._build_shoe()

    def _build_shoe(self):
        """Build and shuffle the shoe; reset running count."""
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
        """Draw a card and update running count. Mark shoe to be reshuffled between hands when cut reached."""
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
        """Call this between hands. If cut card passed, rebuild shoe before dealing."""
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
    """Dealer draws until sum >= 17 (soft 17 stands in this implementation)."""
    while sum_hand(hand) < 17:
        hand.append(draw_card(shoe))
    return hand

def score(player_hand, dealer_hand):
    """Player perspective: +1 win, 0 push, -1 loss."""
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
    return State(player_sum=sum_hand(player_hand),
                 usable_ace=usable_ace(player_hand),
                 dealer_upcard=dealer_upcard,
                 tc_bucket=bucket)


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
    """
    Generate one episode using given policy_fn(state) -> action.
    Returns list of (state, action, reward).
    Uses provided shoe (default global SHOE).
    """
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
                r = -1
                episode.append((s, a, r))
                return episode
            else:
                episode.append((s, a, 0))
        else:  # stand
            dealer = dealer_play(dealer, shoe)
            r = score(player, dealer)
            episode.append((s, a, r))
            return episode

        first_decision = False

def train_mc_onpolicy(num_episodes=TRAIN_EPISODES, epsilon=0.1, exploring_starts=False, print_every=50_000, shoe=None):
    """
    On-policy first-visit MC with epsilon-greedy improvement.
    optimistic initialization: small positive initial Q to avoid early standing bias.
    """
    if shoe is None:
        shoe = SHOE

    # optimistic initialization (small positive to encourage exploration)
    def q_init():
        return {"hit": 0.1, "stand": 0.1}

    Q = defaultdict(lambda: q_init())
    N = defaultdict(lambda: {"hit": 0, "stand": 0})

    def policy_fn(state):
        return epsilon_greedy_action(Q, state, epsilon)

    for ep in range(1, num_episodes + 1):
        episode = generate_episode(policy_fn, exploring_starts=exploring_starts, shoe=shoe)

        final_reward = episode[-1][2]
        visited = set()
        for (s, a, _) in episode:
            key = (s, a)
            if key in visited:
                continue
            visited.add(key)
            N[s][a] += 1
            alpha = 1.0 / N[s][a]
            Q[s][a] += alpha * (final_reward - Q[s][a])

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

def play_episode_with_policy(policy_fn, shoe=None, seed=None):
    """Play one hand using policy_fn on the provided shoe (or global)."""
    if seed is not None:
        random.seed(seed)
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
            continue
        else:
            dealer = dealer_play(dealer, shoe)
            return score(player, dealer)

def evaluate_policy(policy_fn, num_games=10000, num_decks=NUM_DECKS, reshuffle_cut=RESHUFFLE_CUT, seed=None):
    """
    Evaluate a policy on a fresh shoe (isolated from training).
    This swaps in a temporary shoe for the duration of evaluation to avoid contamination.
    """
    if seed is not None:
        random.seed(seed)

    eval_shoe = Shoe(num_decks=num_decks, reshuffle_cut=reshuffle_cut)

    wins = ties = losses = total_reward = 0
    for _ in range(num_games):
        r = play_episode_with_policy(policy_fn, shoe=eval_shoe)
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
        "push_rate": ties / num_games,
        "loss_rate": losses / num_games,
        "average_reward": total_reward / num_games
    }


def play_interactive_game(learned_policy):
    """
    Interactive terminal blackjack game.
    User plays against the dealer with AI recommendations.
    """
    print("\n" + "="*60)
    print("WELCOME TO BLACKJACK - AI ADVISOR MODE")
    print("="*60)
    print("\nThe AI will recommend moves, but YOU decide!")
    print("Card counting is enabled - watch the true count.\n")

    shoe = Shoe(num_decks=NUM_DECKS, reshuffle_cut=RESHUFFLE_CUT)
    total_hands = 0
    wins = 0
    losses = 0
    pushes = 0

    while True:
        shoe.start_new_hand()
        player = draw_hand(shoe)
        dealer = draw_hand(shoe)
        dealer_up = dealer[0]

        total_hands += 1
        print("\n" + "="*60)
        print(f"HAND #{total_hands}")
        print("="*60)
        print(f"Dealer shows: {dealer_up}")
        print(f"Your hand: {player} -> Sum: {sum_hand(player)}")
        if usable_ace(player):
            print("   (You have a usable Ace!)")

        # Show count
        tc = shoe.true_count()
        print(f"True Count: {tc:.2f} | Decks remaining: {shoe.decks_remaining():.1f}")

        # Get AI recommendation
        state = state_from_hands(player, dealer_up, shoe)
        ai_rec = learned_policy(state)
        print(f"AI recommends: {ai_rec.upper()}")

        # Player's turn
        player_busted = False
        while True:
            choice = input("\nYour move - [H]it or [S]tand? ").strip().lower()

            if choice not in ['h', 's', 'hit', 'stand']:
                print("Invalid input! Please enter H or S.")
                continue

            action = 'hit' if choice in ['h', 'hit'] else 'stand'

            if action == 'hit':
                card = draw_card(shoe)
                player.append(card)
                print(f"\nYou drew: {card}")
                print(f"Your hand: {player} -> Sum: {sum_hand(player)}")

                if is_bust(player):
                    print("\nBUST! You lose this hand.")
                    player_busted = True
                    losses += 1
                    break

                # Get new AI recommendation for updated hand
                state = state_from_hands(player, dealer_up, shoe)
                ai_rec = learned_policy(state)
                print(f"AI now recommends: {ai_rec.upper()}")

            else:  # stand
                print("\nYou stand.")
                break

        # Dealer's turn (if player didn't bust)
        if not player_busted:
            print(f"\nDealer reveals: {dealer} -> Sum: {sum_hand(dealer)}")
            dealer = dealer_play(dealer, shoe)
            print(f"Dealer's final hand: {dealer} -> Sum: {sum_hand(dealer)}")

            if is_bust(dealer):
                print("Dealer busts!")

            # Determine winner
            result = score(player, dealer)
            player_sum = sum_hand(player)
            dealer_sum = sum_hand(dealer)

            print(f"\nFinal: You {player_sum} vs Dealer {dealer_sum}")

            if result == 1:
                print("YOU WIN!")
                wins += 1
            elif result == 0:
                print("PUSH (tie)")
                pushes += 1
            else:
                print("Dealer wins.")
                losses += 1

        # Show running statistics
        print(f"\nSession Stats: {wins}W - {losses}L - {pushes}P ({total_hands} hands)")
        if total_hands > 0:
            win_rate = wins / total_hands * 100
            print(f"   Win Rate: {win_rate:.1f}%")

        # Ask to continue
        play_again = input("\nPlay another hand? [Y/n]: ").strip().lower()
        if play_again == 'n' or play_again == 'no':
            print("\n" + "="*60)
            print("Thanks for playing!")
            print(f"Final Stats: {wins}W - {losses}L - {pushes}P")
            print("="*60)
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
        play_interactive_game(learned_policy)

    else:
        print("Invalid choice. Exiting.")
