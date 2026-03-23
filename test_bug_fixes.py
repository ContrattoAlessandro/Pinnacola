import os
import torch
import numpy as np
from pinnacola_env import PinnacolaEnv, ActionType
from custom_avn_train import ValueNet, select_action, NStepReplayBuffer

def test_deadlock():
    print("Testing for deadlock...")
    env = PinnacolaEnv(auto_simulate_opponents=True)
    obs_dim = env.observation_space['observation'].shape[0]
    device = torch.device("cpu")
    policy_net = ValueNet(obs_dim).to(device)
    
    opponent_net = ValueNet(obs_dim).to(device)
    
    def opponent_policy(env_obj, player_id):
        # Usiamo max_actions=10 per simulare lo stesso carico
        return select_action(env_obj, opponent_net, epsilon=0.05, device=device, max_actions=10)
    
    env.opponent_policy_fn = opponent_policy
    
    episodes = 0
    total_steps = 0
    
    while episodes < 500:
        obs, _ = env.reset()
        done = False
        ep_steps = 0
        
        while not done and ep_steps < 500:
            if ep_steps % 50 == 0:
                print(f"Ep {episodes}, Step {ep_steps}")
                
            action_tuple = select_action(env, policy_net, 0.05, device, max_actions=10)
            _, _, terminated, truncated, _ = env.step(np.array(action_tuple))
            ep_steps += 1
            total_steps += 1
            
            if terminated or truncated:
                done = True
        
        episodes += 1
        print(f"Episode {episodes} finished. Steps: {ep_steps}")

if __name__ == "__main__":
    test_deadlock()
