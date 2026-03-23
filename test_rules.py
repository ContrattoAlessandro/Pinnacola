import numpy as np
import torch
from pinnacola_env import PinnacolaEnv, Card, CardRank, CardSuit, ActionType

def test_set_suits():
    print("--- Test 1: Tris con semi uguali ---")
    env = PinnacolaEnv()
    obs, _ = env.reset()
    
    # Crea una mano con un tris non valido (3 carte dello stesso rank e *stesso seme*)
    # Nota: in 2 mazzi ci sono due 5 di cuori, quindi max 2 copie esatte,
    # ma proviamo 2 copie di 5Cuori e una di 5Quadri.
    # In un vero Pinnacola le carte del tris/quartetto devono avere *tutte* semi diversi.
    hand = [
        Card(CardRank.FIVE, CardSuit.HEARTS, 0),
        Card(CardRank.FIVE, CardSuit.HEARTS, 1),
        Card(CardRank.FIVE, CardSuit.DIAMONDS, 0)
    ]
    env.player_hands[0] = hand.copy()
    
    valid_melds = env._find_valid_melds(hand)
    print(f"Mano: {hand}")
    print(f"Combinazioni valide trovate: {valid_melds}")
    if len(valid_melds) > 0:
        print("❌ BUG: Ha considerato valido un tris con semi duplicati (5♥, 5♥, 5♦).")
    else:
        print("✅ CORRETTO: Tris con semi duplicati ignorato.")

def test_action_override():
    print("\n--- Test 2: MELD_SET gioca sempre valid_melds[0] ---")
    env = PinnacolaEnv()
    obs, _ = env.reset()
    env.game_phase = 1 # MELD phase
    
    # Mano con DUE tris validi distinti
    hand = [
        Card(CardRank.SEVEN, CardSuit.HEARTS, 0),
        Card(CardRank.SEVEN, CardSuit.DIAMONDS, 0),
        Card(CardRank.SEVEN, CardSuit.CLUBS, 0),
        Card(CardRank.NINE, CardSuit.HEARTS, 0),
        Card(CardRank.NINE, CardSuit.DIAMONDS, 0),
        Card(CardRank.NINE, CardSuit.SPADES, 0),
    ]
    env.player_hands[0] = hand.copy()
    
    valid_melds = env._find_valid_melds(hand)
    print(f"Combinazioni valide disponibili: {valid_melds}")
    
    # Provo a giocare il SECONDO tris (i tre 9) passando la carta 9 come parametro
    card_9_idx = env.card_to_idx[hand[3]]
    action = np.array([ActionType.MELD_SET, card_9_idx, 0, 3])
    
    print(f"Eseguo azione MELD_SET per la carta: {hand[3]}")
    env.step(action)
    
    print(f"Giochi sul tavolo dopo l'azione: {env.table_melds}")
    # Se ha calato i 7 invece dei 9, è un bug.
    if len(env.table_melds) > 0 and env.table_melds[0].cards[0].rank == CardRank.SEVEN:
         print("❌ BUG: Ho chiesto di giocare i 9, ma ha giocato i 7 (il primo in list).")
    else:
         print("✅ CORRETTO: Ha giocato i 9.")

def test_scale_generation():
    print("\n--- Test 3: Generazione scale (subset mancanti?) ---")
    env = PinnacolaEnv()
    env.reset()
    
    hand = [
        Card(CardRank.THREE, CardSuit.HEARTS, 0),
        Card(CardRank.FOUR, CardSuit.HEARTS, 0),
        Card(CardRank.FIVE, CardSuit.HEARTS, 0),
        Card(CardRank.SIX, CardSuit.HEARTS, 0),
        Card(CardRank.SEVEN, CardSuit.HEARTS, 0),
    ]
    env.player_hands[0] = hand.copy()
    
    valid_melds = env._find_valid_melds(hand)
    print("Scala completa (5 carte): 3, 4, 5, 6, 7")
    print("Combinazioni generate:")
    for i, m in enumerate(valid_melds):
        print(f"  {i+1}) {m}")
        
    # Dovrebbe generare:
    # 3-4-5-6
    # 3-4-5-6-7
    # 4-5-6-7
    # E che dire di 3-4-5? In molti regolamenti di Pinnacola le scale partono da 3 carte.
    lengths = [len(m) for m in valid_melds]
    if 3 not in lengths:
        print("❓ NOTA: Nessuna scala di 3 carte generata. Le scale minime per Pinnacola di solito sono 3 carte, non 4.")

def run_tests():
    test_set_suits()
    test_action_override()
    test_scale_generation()

if __name__ == '__main__':
    run_tests()
