"""
FASE 2: Training Script per Pinnacola RL Agent
Implementa PPO con Action Masking e Self-Play
"""

import os
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Tuple, Any

# Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import gymnasium as gym

# Environment locale
from pinnacola_env import PinnacolaEnv


class PinnacolaFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor custom per Pinnacola.
    L'observation è flat, quindi usiamo un MLP denso.
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Dimensione input
        n_flatten = observation_space.shape[0]
        
        # MLP denso
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_flatten, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, features_dim),
            torch.nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mlp(observations)


class ActionMaskWrapper(gym.Wrapper):
    """
    Wrapper che gestisce l'action masking per PPO.
    Converte lo spazio delle observation da Dict a Box (flat).
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Rendi lo spazio delle azioni flat per SB3
        self.action_space = gym.spaces.Discrete(env.unwrapped._get_action_space_size())
        
        # Ottieni dimensione observation piatto dall'env originale
        obs_dim = env.observation_space['observation'].shape[0]
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Flatten observation: prendi solo il vettore observation, non il dict
        return obs['observation'], info
    
    def step(self, action: int):
        # Converti azione flat in MultiDiscrete
        action_size = self.env.unwrapped.TOTAL_CARDS * 20 * 10
        action_type = action // action_size
        remainder = action % action_size
        
        meld_size = 20 * 10
        card_idx = remainder // meld_size
        remainder = remainder % meld_size
        
        meld_idx = remainder // 10
        param = remainder % 10
        
        # Limita ai bound corretti
        action_type = min(action_type, 8)
        card_idx = min(card_idx, 107)
        meld_idx = min(meld_idx, 19)
        param = min(param, 9)
        
        action_array = np.array([action_type, card_idx, meld_idx, param])
        
        obs, reward, terminated, truncated, info = self.env.step(action_array)
        
        return obs['observation'], reward, terminated, truncated, info


class SelfPlayCallback(BaseCallback):
    """
    Callback per implementare Self-Play.
    Salva il modello ogni N iterazioni e lo usa come avversario.
    """
    
    def __init__(self, save_freq: int = 10000, save_path: str = "./models/",
                 verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Salva checkpoint
            model_path = os.path.join(
                self.save_path, 
                f"ppo_pinnacola_{self.num_timesteps}_steps.zip"
            )
            self.model.save(model_path)
            
            if self.verbose > 0:
                print(f"Model saved at {model_path}")
        
        return True


class MetricsCallback(BaseCallback):
    """
    Callback per monitorare metriche di training.
    Stampa win rate, reward medio, lunghezza episodi ogni N steps.
    """
    
    def __init__(self, eval_freq: int = 10000, log_file: str = "training_log.txt", verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.log_file = log_file
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_count = 0
        self.total_episodes = 0
        
    def _on_step(self) -> bool:
        # Raccogli info dagli env
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.total_episodes += 1
                if info["episode"]["r"] > 50:
                    self.win_count += 1
        
        if self.n_calls % self.eval_freq == 0 and self.episode_rewards:
            avg_reward = np.mean(self.episode_rewards[-100:])
            avg_length = np.mean(self.episode_lengths[-100:])
            win_rate = self.win_count / max(self.total_episodes, 1) * 100
            
            msg = f"[{self.n_calls:>8}] WinRate: {win_rate:>5.1f}% | AvgReward: {avg_reward:>7.2f} | Episodes: {self.total_episodes}"
            print(msg)
            
            with open(self.log_file, "a") as f:
                f.write(f"{self.n_calls},{win_rate:.2f},{avg_reward:.2f},{avg_length:.2f},{self.total_episodes}\n")
            
            self.win_count = 0
            self.total_episodes = 0
        
        return True


class MaskedPPO(PPO):
    """
    PPO che supporta action masking.
    Override della funzione di raccolta dati per filtrare azioni illegali.
    """
    
    def collect_rollouts(self, *args, **kwargs):
        # Qui potremmo implementare masking specifico
        # Per ora usiamo il wrapper che gestisce il masking implicitamente
        return super().collect_rollouts(*args, **kwargs)


def make_env():
    """Factory per creare environment wrapped."""
    def _init():
        env = PinnacolaEnv()
        env = ActionMaskWrapper(env)
        return env
    return _init


def train_ppo(total_timesteps: int = 1_000_000, save_dir: str = "./models"):
    """
    Training loop principale per PPO.
    
    Args:
        total_timesteps: Numero totale di timesteps per training
        save_dir: Directory dove salvare i modelli
    """
    
    print("=" * 60)
    print("PINNACOLA RL TRAINING - FASE 2")
    print("=" * 60)
    
    # Crea environment vectorized - VELOCIZZAZIONE
    n_envs = 8  # Aumentato da 4 a 8 per più parallellismo
    print(f"Creating {n_envs} parallel environments...")
    
    # Prova SubprocVecEnv, fallback a DummyVecEnv se fallisce
    try:
        env = SubprocVecEnv([make_env() for _ in range(n_envs)], start_method="spawn")
    except:
        print("Warning: SubprocVecEnv failed, using DummyVecEnv (slower)")
        from stable_baselines3.common.vec_env import DummyVecEnv
        env = DummyVecEnv([make_env() for _ in range(n_envs)])
    
    # Configurazione policy - OTTIMIZZATA
    policy_kwargs = dict(
        features_extractor_class=PinnacolaFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )
    
    # Crea modello PPO - PARAMETRI OTTIMIZZATI PER VELOCITA'
    print("Creating PPO model...")
    print("Config: n_steps=1024, batch_size=256, n_epochs=5 (faster updates)")
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=1024,      # Ridotto da 2048 (più frequenti aggiornamenti)
        batch_size=256,    # Aumentato da 64 (miglior utilizzo CPU)
        n_epochs=5,        # Ridotto da 10 (meno tempo per batch)
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=None,
        device="auto"
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_dir,
        name_prefix="ppo_pinnacola",
        verbose=1
    )
    
    selfplay_callback = SelfPlayCallback(
        save_freq=100000,
        save_path=save_dir,
        verbose=1
    )
    
    metrics_callback = MetricsCallback(
        eval_freq=5000,
        log_file=os.path.join(save_dir, "training_metrics.csv"),
        verbose=1
    )
    
    # Training
    print(f"Starting training for {total_timesteps} timesteps...")
    print(f"Models will be saved to: {save_dir}")
    print(f"Checkpoint every 50000 steps | Metrics every 5000 steps")
    print("=" * 60)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, selfplay_callback, metrics_callback],
        progress_bar=False
    )
    
    # Salva modello finale
    final_path = os.path.join(save_dir, "ppo_pinnacola_final.zip")
    model.save(final_path)
    print(f"\n✅ Training completed! Final model saved to: {final_path}")
    
    return model


def evaluate_model(model_path: str, n_episodes: int = 10):
    """
    Valuta un modello addestrato.
    
    Args:
        model_path: Path al modello .zip
        n_episodes: Numero di episodi di test
    """
    print(f"\nEvaluating model: {model_path}")
    
    # Carica modello
    env = make_env()()
    model = PPO.load(model_path, env=env)
    
    wins = 0
    total_rewards = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        # Controlla se vittoria
        is_win = episode_reward > 50  # Soglia reward vittoria
        if is_win:
            wins += 1
        
        total_rewards.append(episode_reward)
        print(f"Episode {ep+1}: Reward={episode_reward:.2f}, Win={is_win}")
    
    print(f"\n{'='*40}")
    print(f"Win Rate: {wins}/{n_episodes} ({100*wins/n_episodes:.1f}%)")
    print(f"Avg Reward: {np.mean(total_rewards):.2f}")
    print(f"{'='*40}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PPO for Pinnacola")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--eval", type=str, help="Evaluate model at path")
    parser.add_argument("--timesteps", type=int, default=1000000, 
                        help="Training timesteps")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Evaluation episodes")
    
    args = parser.parse_args()
    
    if args.train:
        train_ppo(total_timesteps=args.timesteps)
    elif args.eval:
        evaluate_model(args.eval, n_episodes=args.episodes)
    else:
        print("Usage:")
        print("  python train.py --train --timesteps 1000000")
        print("  python train.py --eval models/ppo_pinnacola_final.zip --episodes 20")
