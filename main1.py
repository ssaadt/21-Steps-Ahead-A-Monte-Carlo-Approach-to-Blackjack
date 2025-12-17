
import random
from collections import defaultdict, namedtuple
import math



# State: (player_sum, usable_ace (bool), dealer_upcard)
State = namedtuple("State", ["player_sum", "usable_ace", "dealer_upcard"])
ACTIONS = ["hit", "stand"]


def draw_card():
    """Draw a single card (1-10); face cards count as 10, ace represented as 1."""
    # Cards 2-9 with normal probability, 10 covering 10,J,Q,K (4/13), Ace = 1
    card = random.randint(1,13)
    if card >= 10:
        return 10
    return card  # 1..9 where 1 is Ace

def draw_hand():
    return [draw_card(), draw_card()]

def usable_ace(hand):
    """Return True if hand has an ace counted as 11 (i.e., usable ace)."""
    return 1 in hand and sum(hand) + 10 <= 21

def sum_hand(hand):
    s = sum(hand)
    if 1 in hand and s + 10 <= 21:
        return s + 10
    return s

def is_bust(hand):
    return sum_hand(hand) > 21

def dealer_play(hand):
    """Dealer draws until sum >= 17. Dealer treats usable ace as 11."""
    while True:
        if sum_hand(hand) < 17:
            hand.append(draw_card())
            continue
        else:
            break
    return hand

def score(player_hand, dealer_hand):
    """Return game outcome from player's perspective: +1 win, 0 tie, -1 loss."""
    player_sum = sum_hand(player_hand)
    dealer_sum = sum_hand(dealer_hand)
    if player_sum > 21:
        return -1
    if dealer_sum > 21:
        return 1
    if player_sum > dealer_sum:
        return 1
    if player_sum < dealer_sum:
        return -1
    return 0

def state_from_hands(player_hand, dealer_upcard):
    return State(player_sum=sum_hand(player_hand),
                 usable_ace=usable_ace(player_hand),
                 dealer_upcard=dealer_upcard)

def greedy_action(Q, state):
    """Return argmax action given Q (ties broken randomly)."""
    q_hit = Q[state]["hit"]
    q_stand = Q[state]["stand"]
    if q_hit > q_stand:
        return "hit"
    if q_stand > q_hit:
        return "stand"
    return random.choice(ACTIONS)

def epsilon_greedy_action(Q, state, epsilon):
    """Epsilon-greedy action selection."""
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return greedy_action(Q, state)


def generate_episode(policy_fn, exploring_starts=False):
    """
    Generate an episode following policy_fn(state) -> action.
    Return list of (state, action, reward) with final reward only at end (reward repeated).
    If exploring_starts True, start from a random (state, action) by dealing until the player has >= 12.
    """
    # initial deal
    player = draw_hand()
    dealer = draw_hand()
    dealer_up = dealer[0]

    # If player's initial sum < 12, auto-hit until >= 12    (common optimization)
    # We will still include those states (but many implementations ignore <12 states).
    episode = []
    while True:
        s = state_from_hands(player, dealer_up)
        a = policy_fn(s)
        if a == "hit":
            player.append(draw_card())
            if is_bust(player):
                # terminal
                r = -1
                episode.append((s, a, r))
                return episode
            else:
                episode.append((s, a, 0))  # interim reward
                continue
        else:  # stand
            # dealer plays out
            dealer = dealer_play(dealer)
            r = score(player, dealer)
            episode.append((s, a, r))
            return episode


def train_mc_onpolicy(num_episodes=1_000_000, epsilon=0.1, gamma=1.0, print_every=50_000):
    """
    Train an on-policy Monte Carlo agent with epsilon-greedy improvement.
    Returns Q (defaultdict(dict)) and policy function (closure).
    We treat all unvisited state-actions as 0 initially.
    """
    # Q[state][action] -> value
    Q = defaultdict(lambda: {"hit": 0.0, "stand": 0.0})
    N = defaultdict(lambda: {"hit": 0, "stand": 0})  # counts for incremental updates

    def policy_fn(state):
        return epsilon_greedy_action(Q, state, epsilon)

    for ep in range(1, num_episodes + 1):
        episode = generate_episode(policy_fn)

        # First-visit Monte Carlo incremental update
        # For this episodic game with reward only at end, we can update each (s,a) by final reward.
        final_reward = episode[-1][2]
        visited = set()
        for (s, a, _) in episode:
            key = (s, a)
            if key in visited:
                continue
            visited.add(key)
            N[s][a] += 1
            # incremental average
            alpha = 1.0 / N[s][a]
            Q[s][a] += alpha * (final_reward - Q[s][a])

        if ep % print_every == 0:
            print(f"Trained {ep}/{num_episodes} episodes...")

    # Derived deterministic policy (greedy w.r.t Q)
    def learned_policy(state):
        # If state was never seen, choose hit for small sums < 12, otherwise random choice
        if state not in Q:
            # fallback: if player_sum < 17 then hit else stand
            return "hit" if state.player_sum < 17 else "stand"
        return greedy_action(Q, state)

    return Q, learned_policy

def naive_baseline_policy(state):
    """Simple baseline: hit if player_sum < 17 else stand."""
    return "hit" if state.player_sum < 17 else "stand"

def random_policy(state):
    return random.choice(ACTIONS)


def play_episode_with_policy(policy_fn, seed=None):
    """Play one full hand using policy_fn; return reward (1,0,-1)."""
    if seed is not None:
        random.seed(seed)
    player = draw_hand()
    dealer = draw_hand()
    dealer_up = dealer[0]

    while True:
        s = state_from_hands(player, dealer_up)
        a = policy_fn(s)
        if a == "hit":
            player.append(draw_card())
            if is_bust(player):
                return -1
            continue
        else:
            dealer = dealer_play(dealer)
            return score(player, dealer)

def evaluate_policy(policy_fn, num_games=10000):
    wins = 0
    ties = 0
    losses = 0
    total_reward = 0
    for _ in range(num_games):
        r = play_episode_with_policy(policy_fn)
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

if __name__ == "__main__":
    import time

    # Training parameters - adjust as desired
    TRAIN_EPISODES = 1_000_000     # increase for better learning (e.g., 500k+)
    EPSILON = 0.1
    EVAL_GAMES = 50_000

    print("Training Monte Carlo agent...")
    t0 = time.time()
    Q, learned_policy = train_mc_onpolicy(num_episodes=TRAIN_EPISODES, epsilon=EPSILON, print_every=50_000)
    t1 = time.time()
    print(f"Training done in {t1 - t0:.1f}s")

    print("\nEvaluating learned policy...")
    results_learned = evaluate_policy(learned_policy, num_games=EVAL_GAMES)
    print("Learned policy:", results_learned)

    print("\nEvaluating naive baseline (hit < 17)...")
    results_baseline = evaluate_policy(naive_baseline_policy, num_games=EVAL_GAMES)
    print("Baseline policy:", results_baseline)

    print("\nEvaluating random policy...")
    results_random = evaluate_policy(random_policy, num_games=EVAL_GAMES)
    print("Random policy:", results_random)

    # Quick policy inspection: print some actions for typical states
    print("\nSample policy (player_sum, usable_ace, dealer_upcard) -> action (learned)")
    samples = [
        State(12, False, 2), State(12, False, 10),
        State(16, False, 10), State(18, False, 9),
        State(13, True, 6), State(20, False, 10)
    ]
    for s in samples:
        print(s, "->", learned_policy(s))
