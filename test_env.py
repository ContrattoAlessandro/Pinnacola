"""
Test di base per l'environment Pinnacola
Verifica che il gioco possa essere inizializzato e giocato con azioni random
"""

import numpy as np
from pinnacola_env import PinnacolaEnv, ActionType

def test_basic_gameplay():
    """Test che una partita completa possa essere giocata."""
    print("Testing Pinnacola Environment...")
    
    env = PinnacolaEnv()
    obs, info = env.reset(seed=42)
    
    print(f"\nObservation shape: {obs['observation'].shape}")
    print(f"Action mask shape: {obs['action_mask'].shape}")
    print(f"Legal actions count: {np.sum(obs['action_mask'])}")
    
    env.render()
    
    total_reward = 0
    step_count = 0
    max_steps = 200
    
    while step_count < max_steps:
        # Trova azioni legali
        legal_indices = np.where(obs['action_mask'] == 1)[0]
        
        if len(legal_indices) == 0:
            print("No legal actions! Breaking.")
            break
        
        # Scegli azione casuale legale
        action_idx = np.random.choice(legal_indices)
        
        # Decodifica l'azione flat
        action_type = action_idx // (108 * 20 * 10)
        remainder = action_idx % (108 * 20 * 10)
        card_idx = remainder // (20 * 10)
        remainder = remainder % (20 * 10)
        meld_idx = remainder // 10
        param = remainder % 10
        
        action = np.array([action_type, card_idx, meld_idx, param])
        
        # Esegui azione
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        action_name = list(ActionType)[min(action_type, len(ActionType)-1)].name
        print(f"Step {step_count}: {action_name}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
        
        if step_count % 10 == 0:
            env.render()
        
        if terminated or truncated:
            print(f"\nGame ended after {step_count} steps!")
            print(f"Total reward: {total_reward}")
            env.render()
            break
    
    print("\nTest completed successfully!")
    return True

def test_specific_actions():
    """Test azioni specifiche."""
    print("\nTesting specific actions...")
    
    env = PinnacolaEnv()
    obs, info = env.reset(seed=123)
    
    # Test: dovremmo poter pescare dal tallone
    action = np.array([ActionType.DRAW_STOCK, 0, 0, 0])
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"After DRAW_STOCK: reward={reward}, phase changed")
    env.render()
    
    # Test: trova melds nella mano
    hand = env.player_hands[0]
    melds = env._find_valid_melds(hand)
    print(f"Valid melds in hand: {len(melds)}")
    for i, meld in enumerate(melds[:3]):  # Mostra prime 3
        print(f"  Meld {i}: {[str(c) for c in meld]}")
    
    return True

if __name__ == "__main__":
    try:
        test_basic_gameplay()
        test_specific_actions()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
