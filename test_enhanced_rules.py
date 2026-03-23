import numpy as np
from pinnacola_env import PinnacolaEnv, Card, CardRank, CardSuit, ActionType, GamePhase

env = PinnacolaEnv()
env.reset()

# Test 1: _evaluate_meld_points (Poker Assi)
cards = [
    Card(CardRank.ACE, CardSuit.HEARTS, 0),
    Card(CardRank.ACE, CardSuit.DIAMONDS, 0),
    Card(CardRank.ACE, CardSuit.CLUBS, 0),
    Card(CardRank.ACE, CardSuit.SPADES, 0),
]
score = env._evaluate_meld_points(cards, is_set=True)
print(f"Poker Assi Score: {score} (Atteso: 120)")
assert score == 120

# Test 2: _evaluate_meld_points (Poker < 6)
cards2 = [
    Card(CardRank.TWO, CardSuit.HEARTS, 0),
    Card(CardRank.TWO, CardSuit.DIAMONDS, 0),
    Card(CardRank.TWO, CardSuit.CLUBS, 0),
    Card(CardRank.TWO, CardSuit.SPADES, 0),
]
score2 = env._evaluate_meld_points(cards2, is_set=True)
print(f"Poker < 6 Score: {score2} (Atteso: 40)")
assert score2 == 40

# Test 3: _evaluate_meld_points (Scala >= 7)
cards3 = [
    Card(CardRank.THREE, CardSuit.HEARTS, 0),
    Card(CardRank.FOUR, CardSuit.HEARTS, 0), # 10
    Card(CardRank.FIVE, CardSuit.HEARTS, 0), # 15
    Card(CardRank.SIX, CardSuit.HEARTS, 0),  # 25
    Card(CardRank.SEVEN, CardSuit.HEARTS, 0), # 35
    Card(CardRank.EIGHT, CardSuit.HEARTS, 0), # 45
    Card(CardRank.NINE, CardSuit.HEARTS, 0),  # 55
]
score3 = env._evaluate_meld_points(cards3, is_set=False)
print(f"Scala di 7 (+6) Score base: 55 -> * 2: {score3} (Atteso: 110)")
assert score3 == 110

# Test 4: _evaluate_meld_points (Scala < 7)
cards4 = [
    Card(CardRank.THREE, CardSuit.HEARTS, 0),
    Card(CardRank.FOUR, CardSuit.HEARTS, 0), 
    Card(CardRank.FIVE, CardSuit.HEARTS, 0), 
]
score4 = env._evaluate_meld_points(cards4, is_set=False)
print(f"Scala < 7 Score: {score4} (Atteso: 15)")
assert score4 == 15

print("Tutti i test sulle ricompense calcolate passati con successo!")
