# 21 Steps Ahead: A Monte Carlo Approach to Blackjack

An AI-powered blackjack game that uses Monte Carlo reinforcement learning and Hi-Lo card counting to learn optimal playing strategy. Train an AI on 1 million hands, then play interactively with real-time recommendations and intelligent betting.

## Project Overview

Our project combines reinforcement learning research with an interactive blackjack game. The AI learns optimal blackjack strategy through Monte Carlo methods while incorporating card counting into its decision-making. Players can then use the trained AI as an advisor while playing, with dynamic betting based on the card count.

## Reinforcement Learning

The AI uses Monte Carlo first-visit learning to learn optimal blackjack strategy:

* **Algorithm:** Monte Carlo with epsilon-greedy exploration
* **State Space:** Player hand (4-21), usable ace (yes/no), dealer upcard (A-10), true count bucket (7 levels)
* **Actions:** Hit or Stand
* **Training:** 1 million simulated hands with optimistic initialization
* **Card Counting:** Hi-Lo system (+1 for 2-6, -1 for 10-A, 0 for 7-9)

The AI learns when to hit or stand based on your hand, the dealer's card, and the true count. Over time, it discovers that certain situations favor hitting (low count, weak hand) while others favor standing (high count, dealer bust cards).

## Interactive Gameplay

After training, you can play blackjack with the AI as your advisor. Choose from three difficulty modes:

**Beginner Mode**
* Full AI recommendations with detailed explanations
* Card counting hints (high/low count indicators)
* Strategic reasoning for each decision

**Expert Mode**
* Minimal AI suggestions
* True count and decks remaining displayed
* For experienced players who know basic strategy

**Practice Mode**
* No AI help during play
* Post-hand analysis shows what AI would have done
* Learn by making your own decisions

## Betting & Card Counting

The game features intelligent betting based on card counting:

* **Starting Bankroll:** $1,000
* **Dynamic Betting:** Bet more when the count is high (player advantage)
* **Bet Suggestions:** AI recommends bet size based on true count
* **Bankroll Tracking:** Real-time profit/loss, win rate

**Betting Strategy:**

```
True Count ≤ 1:  Bet $10-20   (Neutral/unfavorable)
True Count 2-3:  Bet $40      (Good count)
True Count 4+:   Bet $60      (Excellent count)
```

When the true count is high, there are more 10s and Aces left in the deck, giving you better odds. The AI adjusts bet sizes accordingly to maximize long-term profit.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ssaadt/21-Steps-Ahead-A-Monte-Carlo-Approach-to-Blackjack.git
cd 21-Steps-Ahead-A-Monte-Carlo-Approach-to-Blackjack

# No dependencies needed - just pure Python!
```

### Usage

**Option 1: Train and See Statistics**

```bash
python main2.py
> Enter choice (1/2): 1
```

Trains the AI and compares it against baseline strategies.

**Option 2: Play Interactive Game**

```bash
python main2.py
> Enter choice (1/2): 2
> Select mode (1/2/3): 1  # Choose difficulty
```

Play blackjack with AI guidance. Pick your mode:
* `1` - Beginner (full explanations)
* `2` - Expert (minimal info)
* `3` - Practice (no AI help)
