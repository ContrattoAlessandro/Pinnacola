"""
Test per verificare i fix dei bug in pinnacola_env.py
"""
import numpy as np
from pinnacola_env import PinnacolaEnv, Card, CardRank, CardSuit, ActionType, Meld

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name}")
        failed += 1

# =============================================================================
# BUG 1: Jolly con hash unici
# =============================================================================
print("\n=== BUG 1: Jolly con hash unici ===")
env = PinnacolaEnv()
jokers = [c for c in env.all_cards if c.rank == CardRank.JOKER]
check("4 Jolly nel mazzo", len(jokers) == 4)

hashes = set(hash(j) for j in jokers)
check("Tutti i Jolly hanno hash distinti", len(hashes) == 4)

indices = set(env.card_to_idx[j] for j in jokers)
check("Tutti i Jolly hanno indici distinti in card_to_idx", len(indices) == 4)

# =============================================================================
# BUG 2: Tris con Jolly rifiuta semi duplicati
# =============================================================================
print("\n=== BUG 2: Tris con Jolly — semi distinti required ===")
env = PinnacolaEnv()
env.reset()

# Mano: 5♥ deck0, 5♥ deck1, JOKER — non è un tris valido (semi duplicati)
j = [c for c in env.all_cards if c.rank == CardRank.JOKER][0]
hand_bad = [
    Card(CardRank.FIVE, CardSuit.HEARTS, 0),
    Card(CardRank.FIVE, CardSuit.HEARTS, 1),
    j,
]
melds_bad = env._find_valid_melds(hand_bad)
tris_bad = [m for m in melds_bad if len(set(c.rank for c in m if c.rank != CardRank.JOKER)) == 1]
check("Tris 5♥+5♥+JOKER rifiutato (semi duplicati)", len(tris_bad) == 0)

# Mano: 5♥, 5♦, JOKER — tris valido (semi diversi)
hand_good = [
    Card(CardRank.FIVE, CardSuit.HEARTS, 0),
    Card(CardRank.FIVE, CardSuit.DIAMONDS, 0),
    j,
]
melds_good = env._find_valid_melds(hand_good)
tris_good = [m for m in melds_good if len(set(c.rank for c in m if c.rank != CardRank.JOKER)) == 1]
check("Tris 5♥+5♦+JOKER accettato (semi diversi)", len(tris_good) > 0)

# Quartetto con Jolly non dovrebbe esistere (4 semi bastano)
hand_q = [
    Card(CardRank.FIVE, CardSuit.HEARTS, 0),
    Card(CardRank.FIVE, CardSuit.DIAMONDS, 0),
    Card(CardRank.FIVE, CardSuit.CLUBS, 0),
    j,
]
melds_q = env._find_valid_melds(hand_q)
quartets_with_joker = [m for m in melds_q if len(m) == 4 and any(c.rank == CardRank.JOKER for c in m)]
check("Nessun quartetto con Jolly (4 semi != 3+Jolly)", len(quartets_with_joker) == 0)

# =============================================================================
# BUG 3: Scale con 2 Jolly
# =============================================================================
print("\n=== BUG 3: Scale con 2 Jolly ===")
j2 = [c for c in env.all_cards if c.rank == CardRank.JOKER][1]
hand_2j = [
    Card(CardRank.THREE, CardSuit.HEARTS, 0),
    Card(CardRank.SIX, CardSuit.HEARTS, 0),
    j, j2,
]
melds_2j = env._find_valid_melds(hand_2j)
runs_2j = [m for m in melds_2j if sum(1 for c in m if c.rank == CardRank.JOKER) == 2]
check("Scala 3♥, JOKER, JOKER, 6♥ generata", len(runs_2j) > 0)

# =============================================================================
# BUG 5: Azioni legali — SET vs RUN distinti
# =============================================================================
print("\n=== BUG 5: Azioni legali — SET vs RUN distinti ===")
env2 = PinnacolaEnv()
env2.reset()
env2.game_phase = 1  # MELD

# Mano con un tris e una scala
env2.player_hands[0] = [
    Card(CardRank.SEVEN, CardSuit.HEARTS, 0),
    Card(CardRank.SEVEN, CardSuit.DIAMONDS, 0),
    Card(CardRank.SEVEN, CardSuit.CLUBS, 0),
    Card(CardRank.THREE, CardSuit.SPADES, 0),
    Card(CardRank.FOUR, CardSuit.SPADES, 0),
    Card(CardRank.FIVE, CardSuit.SPADES, 0),
]

legal = env2._get_legal_actions(0)
meld_set_actions = [a for a in legal if a[0] == ActionType.MELD_SET]
meld_run_actions = [a for a in legal if a[0] == ActionType.MELD_RUN]
check("Azioni MELD_SET presenti", len(meld_set_actions) > 0)
check("Azioni MELD_RUN presenti", len(meld_run_actions) > 0)

# Conta: dovrebbe essere 1 azione per il tris, 1 per la scala (non N per carta)
check("1 azione per tris (non 3)", len(meld_set_actions) == 1)
check("1 azione per scala (non 3)", len(meld_run_actions) == 1)

# =============================================================================
# BUG 6/7: Asso alto — Q-K-A scala valida
# =============================================================================
print("\n=== BUG 6/7: Asso alto ===")
# can_attach: Asso su meld che finisce con K
meld_qk = Meld(meld_id=0, meld_type='run', cards=[
    Card(CardRank.QUEEN, CardSuit.HEARTS, 0),
    Card(CardRank.KING, CardSuit.HEARTS, 0),
])
ace = Card(CardRank.ACE, CardSuit.HEARTS, 0)
check("Asso attaccabile a scala Q-K (stessa suit)", meld_qk.can_attach(ace))

ace_wrong_suit = Card(CardRank.ACE, CardSuit.SPADES, 0)
check("Asso NON attaccabile a scala Q-K (suit diversa)", not meld_qk.can_attach(ace_wrong_suit))

# Scala Q-K-A generata da _find_valid_melds
env3 = PinnacolaEnv()
env3.reset()
hand_ace_high = [
    Card(CardRank.QUEEN, CardSuit.HEARTS, 0),
    Card(CardRank.KING, CardSuit.HEARTS, 0),
    Card(CardRank.ACE, CardSuit.HEARTS, 0),
]
melds_ah = env3._find_valid_melds(hand_ace_high)
check("Scala Q-K-A generata", len(melds_ah) > 0)

# =============================================================================
# BUG 8: Sostituzione Jolly su qualsiasi meld
# =============================================================================
print("\n=== BUG 8: Sostituzione Jolly su qualsiasi meld ===")
env4 = PinnacolaEnv()
env4.reset()
env4.game_phase = 1  # MELD

joker_card = [c for c in env4.all_cards if c.rank == CardRank.JOKER][0]
opponent_meld = Meld(meld_id=0, meld_type='set', cards=[
    Card(CardRank.NINE, CardSuit.HEARTS, 0),
    Card(CardRank.NINE, CardSuit.DIAMONDS, 0),
    joker_card,
], owner=1)  # Posseduto dall'avversario
env4.table_melds = [opponent_meld]

replacement = Card(CardRank.NINE, CardSuit.CLUBS, 0)
env4.player_hands[0] = [replacement]

legal4 = env4._get_legal_actions(0)
replace_actions = [a for a in legal4 if a[0] == ActionType.REPLACE_JOKER]
check("Sostituzione Jolly permessa su meld avversario", len(replace_actions) > 0)

# =============================================================================
# BUG 11: cards_seen aggiornato dagli avversari
# =============================================================================
print("\n=== BUG 11: cards_seen aggiornato quando avversario scarta ===")
env5 = PinnacolaEnv()
env5.reset(seed=42)

# Salva stato carte viste dal bot prima della simulazione avversari
bot_seen_before = env5.cards_seen[0].sum()

# Forza il turno a un avversario e simula
env5.current_player = 1
env5._simulate_opponent_turns()

bot_seen_after = env5.cards_seen[0].sum()
check("cards_seen del bot aumentato dopo scarto avversari", bot_seen_after > bot_seen_before)

# =============================================================================
# Test integrazione: partita completa senza crash
# =============================================================================
print("\n=== Test integrazione: partita completa ===")
env6 = PinnacolaEnv()
obs, _ = env6.reset(seed=123)
steps = 0
total_reward = 0
crashed = False

try:
    while steps < 300:
        legal = env6._get_legal_actions()
        if not legal:
            break
        action = legal[np.random.randint(len(legal))]
        obs, reward, terminated, truncated, _ = env6.step(np.array(action))
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break
except Exception as e:
    crashed = True
    print(f"  💥 CRASH: {e}")

check(f"Partita completata senza crash ({steps} steps)", not crashed)
check(f"Partita terminata normalmente", steps > 0)

# =============================================================================
# Riepilogo
# =============================================================================
print(f"\n{'='*50}")
print(f"RISULTATO: {passed} passati, {failed} falliti su {passed + failed} test")
print(f"{'='*50}")
