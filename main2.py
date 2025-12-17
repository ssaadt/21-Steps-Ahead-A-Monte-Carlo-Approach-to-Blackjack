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


def format_card(card):
    """Format card for display: 1 -> A, others as-is"""
    return 'A' if card == 1 else str(card)

def format_hand(hand):
    """Format a hand for display with aces shown as 'A'"""
    return '[' + ', '.join(format_card(c) for c in hand) + ']'


def play_interactive_game(learned_policy, mode='expert'):
    """
    Interactive terminal blackjack game.
    User plays against the dealer with AI recommendations.

    Modes:
    - beginner: Shows AI suggestions, probabilities, and detailed explanations
    - expert: Shows AI suggestions and true count (default)
    - practice: No AI suggestions, just play
    """
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    mode_title = f"BLACKJACK - {mode.upper()} MODE"
    print("║" + mode_title.center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")

    if mode == 'beginner':
        print("\n┌" + "─"*58 + "┐")
        print("│  BEGINNER MODE: AI will guide you with explanations      │")
        print("│  Card counting hints and strategy tips included          │")
        print("└" + "─"*58 + "┘\n")
    elif mode == 'expert':
        print("\n┌" + "─"*58 + "┐")
        print("│  EXPERT MODE: AI suggestions with card count             │")
        print("│  Minimal information - you know what you're doing!       │")
        print("└" + "─"*58 + "┘\n")
    else:  # practice
        print("\n┌" + "─"*58 + "┐")
        print("│  PRACTICE MODE: No AI help, just play         │")
        print("│  Test your skills and learn from your mistakes!          │")
        print("└" + "─"*58 + "┘\n")

    # Betting setup (practice mode has no betting)
    if mode != 'practice':
        bankroll = 1000
        min_bet = 10
        print(f"  Starting bankroll: ${bankroll}")
        print(f"  Minimum bet: ${min_bet}\n")

    shoe = Shoe(num_decks=NUM_DECKS, reshuffle_cut=RESHUFFLE_CUT)
    total_hands = 0
    wins = 0
    losses = 0
    pushes = 0
    total_wagered = 0
    total_won = 0

    while True:
        shoe.start_new_hand()
        player = draw_hand(shoe)
        dealer = draw_hand(shoe)
        dealer_up = dealer[0]

        total_hands += 1
        print("\n" + "┏" + "━"*58 + "┓")
        print("┃" + f"  HAND #{total_hands}".ljust(58) + "┃")
        print("┗" + "━"*58 + "┛")

        # Get true count before dealing
        tc = shoe.true_count()

        # Betting logic (not for practice mode)
        if mode != 'practice':
            # Check if player has enough money
            if bankroll < min_bet:
                print("\n  ╔" + "═"*40 + "╗")
                print("  ║" + "  OUT OF MONEY!  ".center(40) + "║")
                print("  ║" + f"  Final bankroll: $0".center(40) + "║")
                print("  ╚" + "═"*40 + "╝\n")
                break

            print(f"\n  Bankroll: ${bankroll}")

            # Suggest bet based on true count
            if tc <= 1:
                suggested_bet = min_bet
                bet_reason = "Neutral/negative count"
            elif tc <= 2:
                suggested_bet = min_bet * 2
                bet_reason = "Slightly favorable"
            elif tc <= 3:
                suggested_bet = min_bet * 4
                bet_reason = "Good count!"
            else:
                suggested_bet = min_bet * 6
                bet_reason = "Excellent count!"

            # Don't suggest more than 10% of bankroll
            suggested_bet = min(suggested_bet, bankroll // 10)
            suggested_bet = max(suggested_bet, min_bet)

            if mode == 'beginner':
                print(f"  Suggested bet: ${suggested_bet} ({bet_reason})")
                print(f"  True count: {tc:+.2f}")

            # Get bet from player
            while True:
                bet_input = input(f"\n  Enter bet amount (${min_bet}-${bankroll}): $").strip()
                try:
                    current_bet = int(bet_input)
                    if current_bet < min_bet:
                        print(f"  Minimum bet is ${min_bet}")
                    elif current_bet > bankroll:
                        print(f"  You only have ${bankroll}")
                    else:
                        break
                except ValueError:
                    print("  Please enter a valid number")

            bankroll -= current_bet
            total_wagered += current_bet
            print(f"  ✓ Bet placed: ${current_bet}")

        print(f"\n  Dealer's Card:  {format_card(dealer_up)}")
        print(f"  Your Hand:      {format_hand(player)} = {sum_hand(player)}")
        if usable_ace(player):
            print("                  └─> Soft hand (usable Ace)")

        # Show count info based on mode
        if mode == 'beginner':
            print(f"\n  Count Info:     TC: {tc:+.2f}  |  Decks Left: {shoe.decks_remaining():.1f}")
            if tc > 2:
                print("                  └─> High count! Favorable for player")
            elif tc < -1:
                print("                  └─> Low count! Favorable for dealer")
        elif mode == 'expert':
            print(f"\n  Count Info:     TC: {tc:+.2f}  |  Decks Left: {shoe.decks_remaining():.1f}")
        # Practice mode: no count info

        # Get AI recommendation based on mode
        state = state_from_hands(player, dealer_up, shoe)
        ai_rec = learned_policy(state)

        if mode == 'beginner':
            print(f"\n  AI Suggests:    {ai_rec.upper()}")
            # Add detailed explanation for beginners
            player_total = sum_hand(player)
            has_ace = usable_ace(player)

            print("                  ┌─ Why?")
            if ai_rec == 'hit':
                if player_total <= 11:
                    print("                  │  • Your total is 11 or less - impossible to bust!")
                    print("                  │  • Always hit when you can't go over 21")
                elif player_total == 12:
                    if dealer_up in [4, 5, 6]:
                        print(f"                  │  • Dealer shows {dealer_up} (weak card)")
                        print("                  │  • But 12 is risky, one hit could improve your hand")
                    else:
                        print(f"                  │  • Dealer shows {dealer_up}")
                        print("                  │  • 12 is weak, need to improve your hand")
                elif 13 <= player_total <= 16:
                    if dealer_up >= 7:
                        print(f"                  │  • Dealer shows {dealer_up} (strong card)")
                        print(f"                  │  • Your {player_total} is likely losing")
                        print("                  │  • Must take the risk to improve")
                    else:
                        print(f"                  │  • Dealer shows {dealer_up} (weak upcard)")
                        if tc > 0:
                            print("                  │  • Positive count suggests more 10s left")
                            print("                  │  • Risky but mathematically sound")
                        else:
                            print("                  │  • Strategy says hit against this dealer card")
                elif player_total >= 17:
                    print(f"                  │  • Unusual! Your {player_total} is normally strong")
                    print(f"                  │  • Count is {tc:+.2f} - this affects the decision")
                    print("                  │  • Trust the math, but this is risky!")
            else:  # stand
                if has_ace:
                    print(f"                  │  • Soft {player_total} is strong")
                    print("                  │  • Your Ace gives you flexibility")
                if player_total >= 17:
                    print(f"                  │  • {player_total} is a strong hand")
                    print(f"                  │  • Dealer shows {dealer_up}")
                    if dealer_up <= 6:
                        print("                  │  • Dealer likely to bust with weak upcard")
                    else:
                        print("                  │  • Let dealer try to beat you")
                elif 13 <= player_total <= 16:
                    if dealer_up <= 6:
                        print(f"                  │  • Dealer shows weak card ({dealer_up})")
                        print("                  │  • High chance dealer will bust")
                        print("                  │  • Stand and let them take the risk")
                else:
                    print(f"                  │  • Standing with {player_total}")
                    print("                  │  • Based on count and dealer's card")
            print("                  └─────────────────────────────────")
        elif mode == 'expert':
            print(f"  AI Suggests:    {ai_rec.upper()}")
        # Practice mode: no AI suggestions

        print("  " + "─"*56)

        # Player's turn
        player_busted = False
        while True:
            choice = input("\n  ► Your move - [H]it or [S]tand? ").strip().lower()

            if choice not in ['h', 's', 'hit', 'stand']:
                print("  ✗ Invalid input! Please enter H or S.")
                continue

            action = 'hit' if choice in ['h', 'hit'] else 'stand'

            if action == 'hit':
                card = draw_card(shoe)
                player.append(card)
                print(f"\n  ┌─ Drew: {format_card(card)}")
                print(f"  │  Your Hand: {format_hand(player)} = {sum_hand(player)}")

                if is_bust(player):
                    print(f"  └─ BUST! Over 21")
                    print("\n  ╔" + "═"*30 + "╗")
                    print("  ║" + "  YOU LOSE  ".center(30) + "║")
                    print("  ╚" + "═"*30 + "╝")
                    player_busted = True
                    losses += 1
                    if mode != 'practice':
                        print(f"  Lost ${current_bet}. Bankroll: ${bankroll}")
                    break

                # Get new AI recommendation for updated hand (except practice mode)
                if mode != 'practice':
                    state = state_from_hands(player, dealer_up, shoe)
                    ai_rec = learned_policy(state)
                    if mode == 'beginner':
                        player_total = sum_hand(player)
                        print(f"  └─ AI now suggests: {ai_rec.upper()}")
                        if ai_rec == 'stand':
                            if player_total >= 17:
                                print(f"     • {player_total} is strong - stop here")
                            elif player_total >= 13:
                                print(f"     • {player_total} vs dealer's {dealer_up}")
                                print(f"     • Further hits too risky now")
                            else:
                                print(f"     • Standing pat with {player_total}")
                        else:
                            if player_total <= 11:
                                print(f"     • Still can't bust - keep hitting!")
                            else:
                                print(f"     • {player_total} needs improvement")
                                print(f"     • Risk is acceptable given the situation")
                    else:
                        print(f"  └─ AI now suggests: {ai_rec.upper()}")

            else:  # stand
                print(f"\n  ✓ You stand with {sum_hand(player)}")
                break

        # Dealer's turn (if player didn't bust)
        if not player_busted:
            print(f"\n  " + "─"*56)
            print(f"  Dealer reveals: {format_hand(dealer)} = {sum_hand(dealer)}")
            dealer = dealer_play(dealer, shoe)
            print(f"  Dealer's final: {format_hand(dealer)} = {sum_hand(dealer)}")

            if is_bust(dealer):
                print("  └─ Dealer busts!")

            # Determine winner
            result = score(player, dealer)
            player_sum = sum_hand(player)
            dealer_sum = sum_hand(dealer)

            print(f"\n  ┌─ Final Score ─┐")
            print(f"  │ You:    {str(player_sum).rjust(2)}    │")
            print(f"  │ Dealer: {str(dealer_sum).rjust(2)}    │")
            print(f"  └───────────────┘")

            if result == 1:
                print("\n  ╔" + "═"*30 + "╗")
                print("  ║" + "  YOU WIN!  ".center(30) + "║")
                print("  ╚" + "═"*30 + "╝")
                wins += 1
                if mode != 'practice':
                    payout = current_bet * 2
                    bankroll += payout
                    total_won += payout
                    print(f"  Won ${current_bet}! New bankroll: ${bankroll}")
            elif result == 0:
                print("\n  ╔" + "═"*30 + "╗")
                print("  ║" + "  PUSH (TIE)  ".center(30) + "║")
                print("  ╚" + "═"*30 + "╝")
                pushes += 1
                if mode != 'practice':
                    bankroll += current_bet  # Return bet
                    print(f"  Bet returned. Bankroll: ${bankroll}")
            else:
                print("\n  ╔" + "═"*30 + "╗")
                print("  ║" + "  DEALER WINS  ".center(30) + "║")
                print("  ╚" + "═"*30 + "╝")
                losses += 1
                if mode != 'practice':
                    print(f"  Lost ${current_bet}. Bankroll: ${bankroll}")

            # Post-hand analysis for practice mode
            if mode == 'practice':
                print("\n  ┌─ Post-Hand Analysis ───────────────────────┐")
                # Show what AI recommended for the initial hand
                initial_hand = [player[0], player[1]]
                initial_total = sum(initial_hand) + (10 if 1 in initial_hand and sum(initial_hand) + 10 <= 21 else 0)
                initial_state = state_from_hands(initial_hand, dealer_up, shoe)
                ai_initial = learned_policy(initial_state)

                print(f"  │ Your initial hand: {format_hand(initial_hand)} = {initial_total}")
                print(f"  │ Dealer showed: {format_card(dealer_up)}")
                print(f"  │ ")
                print(f"  │ AI would have: {ai_initial.upper()}")
                print(f"  │ True count was: {tc:+.2f}")

                # Add explanation
                if ai_initial == 'hit':
                    if initial_total <= 11:
                        print(f"  │ Reason: Can't bust with {initial_total}")
                    elif dealer_up >= 7:
                        print(f"  │ Reason: Dealer's {dealer_up} is strong")
                    else:
                        print(f"  │ Reason: Optimal vs dealer's {dealer_up}")
                else:
                    if initial_total >= 17:
                        print(f"  │ Reason: {initial_total} is strong")
                    elif dealer_up <= 6:
                        print(f"  │ Reason: Let dealer bust with weak {dealer_up}")
                    else:
                        print(f"  │ Reason: Risk/reward favors standing")

                print("  └────────────────────────────────────────────┘")

        # Show running statistics
        print(f"\n  ╔═══ Session Stats " + "═"*39 + "╗")
        if total_hands > 0:
            win_rate = wins / total_hands * 100
            stats_line = f"  ║  Hands: {total_hands}  │  W: {wins}  L: {losses}  P: {pushes}  │  Win Rate: {win_rate:.1f}%"
            print(stats_line + " "*(60 - len(stats_line)) + "║")
            if mode != 'practice':
                net_profit = bankroll - 1000
                profit_sign = "+" if net_profit >= 0 else ""
                bankroll_line = f"  ║  Bankroll: ${bankroll}  │  Net: {profit_sign}${net_profit}"
                print(bankroll_line + " "*(60 - len(bankroll_line)) + "║")
        else:
            print(f"  ║  No hands played yet" + " "*37 + "║")
        print(f"  ╚" + "═"*58 + "╝")

        # Ask to continue
        play_again = input("\n  ► Play another hand? [Y/n]: ").strip().lower()
        if play_again == 'n' or play_again == 'no':
            print("\n  " + "╔" + "═"*56 + "╗")
            print("  ║" + "  Thanks for playing!  ".center(56) + "║")
            print("  ║" + " "*56 + "║")
            print("  ║" + f"  Final Stats: {wins}W - {losses}L - {pushes}P  ({total_hands} hands)".center(56) + "║")
            if total_hands > 0:
                print("  ║" + f"  Win Rate: {win_rate:.1f}%".center(56) + "║")
            if mode != 'practice':
                net_profit = bankroll - 1000
                profit_sign = "+" if net_profit >= 0 else ""
                print("  ║" + " "*56 + "║")
                print("  ║" + f"  Final Bankroll: ${bankroll}".center(56) + "║")
                print("  ║" + f"  Net Profit: {profit_sign}${net_profit}".center(56) + "║")
            print("  ╚" + "═"*56 + "╝\n")
            break


if __name__ == "__main__":
    EPSILON = 0.1
    EXPLORING_STARTS = True
    EVAL_GAMES = 50_000

    # Ask user what they want to do
    print("\n╔" + "═"*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "BLACKJACK MONTE CARLO AI".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "═"*58 + "╝")
    print("\n┌─ What would you like to do? " + "─"*28 + "┐")
    print("│                                                          │")
    print("│  [1] Train AI and see statistics                        │")
    print("│  [2] Train AI, then play interactive game               │")
    print("│                                                          │")
    print("└" + "─"*58 + "┘")

    choice = input("\n> Enter choice (1/2): ").strip()

    if choice == "1":
        # Original behavior - train and evaluate
        print("\n┌" + "─"*58 + "┐")
        print("│  Training Monte Carlo AI...                            │")
        print("│  (6-deck shoe, true count tracking)                    │")
        print("└" + "─"*58 + "┘\n")
        t0 = time.time()
        Q, learned_policy = train_mc_onpolicy(num_episodes=TRAIN_EPISODES,
                                              epsilon=EPSILON,
                                              exploring_starts=EXPLORING_STARTS,
                                              print_every=50_000,
                                              shoe=SHOE)
        t1 = time.time()
        print(f"\nTraining completed in {t1 - t0:.1f}s")

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
        print("\n┌" + "─"*58 + "┐")
        print("│  Training Monte Carlo AI...                             │")
        print("│  (6-deck shoe, true count tracking)                     │")
        print("└" + "─"*58 + "┘\n")
        t0 = time.time()
        Q, learned_policy = train_mc_onpolicy(num_episodes=TRAIN_EPISODES,
                                              epsilon=EPSILON,
                                              exploring_starts=EXPLORING_STARTS,
                                              print_every=50_000,
                                              shoe=SHOE)
        t1 = time.time()
        print(f"\nTraining completed in {t1 - t0:.1f}s")
        print("AI is ready to advise!\n")

        # Select difficulty mode
        print("┌─ Select Difficulty Mode " + "─"*32 + "┐")
        print("│                                                          │")
        print("│  [1] Beginner  - AI suggestions with explanations       │")
        print("│  [2] Expert    - AI suggestions, minimal info           │")
        print("│  [3] Practice  - No AI help, learn on your own          │")
        print("│                                                          │")
        print("└" + "─"*58 + "┘")

        mode_choice = input("\n> Select mode (1/2/3): ").strip()

        if mode_choice == "1":
            mode = 'beginner'
        elif mode_choice == "2":
            mode = 'expert'
        elif mode_choice == "3":
            mode = 'practice'
        else:
            print("\nInvalid choice. Defaulting to Expert mode.")
            mode = 'expert'

        # Start interactive game
        play_interactive_game(learned_policy, mode=mode)

    else:
        print("\nInvalid choice. Exiting.\n")
