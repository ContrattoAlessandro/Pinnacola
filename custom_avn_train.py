"""
FASE 3: Custom Afterstate Value Network (AVN) per Pinnacola
Ottimizzato per Mac M4: GPU MPS + State Save/Restore (no deepcopy)
"""

import os
import time
import copy
import random
import numpy as np
from collections import deque
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from pinnacola_env import PinnacolaEnv, Meld

# ============================================================================
# ARCHITETTURA RETE NEURALE: V(s)
# ============================================================================

class ValueNet(nn.Module):
    """MLP per stimare V(s). Output: singolo scalare."""
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# ============================================================================
# REPLAY BUFFER
# ============================================================================

class NStepReplayBuffer:
    """Buffer circolare con accumulo N-Step TD."""
    def __init__(self, capacity=100000, n_step=3, gamma=0.99):
        self.buffer = deque(maxlen=capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)
        
    def push(self, state, reward, next_state, done):
        self.n_step_buffer.append((state, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step:
            s, r, s_next, d = self._get_n_step_info()
            self.buffer.append((s, r, s_next, d))
            
        if done:
            # Svuota il buffer a fine episodio
            while len(self.n_step_buffer) > 0:
                s, r, s_next, d = self._get_n_step_info()
                self.buffer.append((s, r, s_next, d))
                self.n_step_buffer.popleft()
            self.n_step_buffer.clear()
            
    def _get_n_step_info(self):
        # Il primo stato della sequenza
        state, _, _, _ = self.n_step_buffer[0]
        # L'ultimo stato della sequenza
        _, _, next_state, done = self.n_step_buffer[-1]
        
        # Calcola reward cumulato scontato
        reward = 0
        for i, (_, r, _, d) in enumerate(self.n_step_buffer):
            reward += r * (self.gamma ** i)
            if d and i != len(self.n_step_buffer) - 1:
                # Se c'è un done prima della fine, il next_state e done sono quelli
                _, _, next_state, done = self.n_step_buffer[i]
                break
                
        return state, reward, next_state, done
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
        
    def __len__(self):
        return len(self.buffer)

# ============================================================================
# STATE SAVE/RESTORE (sostituisce deepcopy, ~50x più veloce)
# ============================================================================

def save_env_state(env: PinnacolaEnv) -> dict:
    """Salva lo stato mutabile dell'environment in un dizionario leggero."""
    return {
        'player_hands': [hand[:] for hand in env.player_hands],
        'table_melds': copy.deepcopy(env.table_melds),
        'discard_pile': env.discard_pile[:],
        'stock_pile': env.stock_pile[:],
        'game_phase': env.game_phase,
        'current_player': env.current_player,
        'turn_count': env.turn_count,
        'bot_has_drawn': env.bot_has_drawn,
        'bot_melded_this_turn': env.bot_melded_this_turn,
        'round_over': env.round_over,
        'cards_seen': env.cards_seen.copy(),
        'must_meld_card': copy.deepcopy(env.must_meld_card) if hasattr(env, 'must_meld_card') else None
    }

def restore_env_state(env: PinnacolaEnv, state: dict):
    """Ripristina lo stato dell'environment da un dizionario."""
    env.player_hands = [hand[:] for hand in state['player_hands']]
    env.table_melds = copy.deepcopy(state['table_melds'])
    env.discard_pile = state['discard_pile'][:]
    env.stock_pile = state['stock_pile'][:]
    env.game_phase = state['game_phase']
    env.current_player = state['current_player']
    env.turn_count = state['turn_count']
    env.bot_has_drawn = state['bot_has_drawn']
    env.bot_melded_this_turn = state['bot_melded_this_turn']
    env.round_over = state['round_over']
    env.cards_seen = state['cards_seen'].copy()
    if 'must_meld_card' in state:
        env.must_meld_card = copy.deepcopy(state['must_meld_card'])

# ============================================================================
# DEDUPLICA AZIONI (evita simulare azioni equivalenti)
# ============================================================================

def deduplicate_actions(legal_actions):
    """
    Rimuove azioni duplicate/equivalenti.
    Es: DRAW_STOCK con card_idx diversi sono la stessa azione.
    """
    seen = set()
    unique = []
    for at, card, meld, param in legal_actions:
        # DRAW_STOCK e DRAW_PILE non dipendono dalla carta
        if at == 0:  # Only DRAW_STOCK is truly deduplicatable
            key = (at,)
        # SKIP_MELD e CLOSE_ROUND non hanno parametri
        elif at in (6, 8):
            key = (at,)
        else:
            key = (at, card, meld)
        
        if key not in seen:
            seen.add(key)
            unique.append((at, card, meld, param))
    return unique

# ============================================================================
# AVN ACTION SELECTION (con Save/Restore)
# ============================================================================

def select_action(env: PinnacolaEnv, model: nn.Module, epsilon: float, device: torch.device, max_actions: int = 50):
    """
    Seleziona l'azione valutando gli Afterstates dal punto di vista del giocatore corrente.
    Usa save/restore dello stato invece di deepcopy (~50x più veloce).
    """
    acting_player = env.current_player
    legal_actions = env._get_legal_actions(acting_player)
    if not legal_actions:
        legal_actions = [(6, 0, 0, 0)]
    
    # Deduplica azioni equivalenti
    legal_actions = deduplicate_actions(legal_actions)
    
    # Epsilon-Greedy: esplorazione casuale
    if random.random() < epsilon:
        return random.choice(legal_actions)
    
    # Limita il numero di azioni da valutare (per velocità)
    if len(legal_actions) > max_actions:
        eval_actions = random.sample(legal_actions, max_actions)
    else:
        eval_actions = legal_actions
    
    # Salva stato corrente
    saved = save_env_state(env)
    
    # Disabilita auto simulate per valutare l'afterstate pulito
    was_auto = getattr(env, 'auto_simulate_opponents', False)
    env.auto_simulate_opponents = False
    
    next_states = []
    rewards = []
    dones = []
    
    for action_tuple in eval_actions:
        # Applica azione
        _, reward, terminated, truncated, _ = env.step(np.array(action_tuple))
        
        # Ottieni stato dal punto di vista di chi sta agendo
        obs_array = env._get_observation(acting_player)
        next_states.append(obs_array)
        rewards.append(reward)
        dones.append(terminated or truncated)
        
        # Ripristina lo stato
        restore_env_state(env, saved)
        
    env.auto_simulate_opponents = was_auto
    
    # Batch predict su GPU
    states_tensor = torch.FloatTensor(np.array(next_states)).to(device)
    with torch.no_grad():
        values = model(states_tensor).squeeze(-1).cpu().numpy()
    
    # Scegli azione che massimizza R + γ·V(s')
    gamma = 0.99
    best_idx = 0
    best_value = -float('inf')
    for i in range(len(eval_actions)):
        q_val = rewards[i] + (gamma * values[i] * (1 - dones[i]))
        if q_val > best_value:
            best_value = q_val
            best_idx = i
    
    return eval_actions[best_idx]

# ============================================================================
# DEVICE SETUP
# ============================================================================

def get_device():
    """Seleziona il device migliore disponibile: MPS (Mac M4) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        print("🚀 GPU Metal (MPS) rilevata — training su GPU Apple M4")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("🚀 CUDA GPU rilevata")
        return torch.device("cuda")
    else:
        print("⚠️  Solo CPU disponibile")
        return torch.device("cpu")

# ============================================================================
# EVALUATION OFFLINE
# ============================================================================

def evaluate_current_model(model: nn.Module, device: torch.device, n_episodes: int = 10):
    """Esegue N episodi in modalità 100% greedy per valutare la reale forza dell'agente."""
    eval_env = PinnacolaEnv()
    wins = 0
    total_reward = 0
    
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done, ep_reward, ep_steps = False, 0, 0
        
        while not done and ep_steps < 500:
            action_tuple = select_action(eval_env, model, epsilon=0.0, device=device)
            obs, reward, terminated, truncated, _ = eval_env.step(np.array(action_tuple))
            ep_reward += reward
            ep_steps += 1
            done = terminated or truncated
            
        wins += int(ep_reward > 50)
        total_reward += ep_reward
        
    return (wins / max(1, n_episodes)) * 100, total_reward / max(1, n_episodes)


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_avn(total_timesteps=100_000, save_dir="./models"):
    print("=" * 60)
    print("PINNACOLA RL - Afterstate Value Network (Ottimizzato)")
    print("=" * 60)
    
    os.makedirs(save_dir, exist_ok=True)
    device = get_device()
    
    env = PinnacolaEnv(auto_simulate_opponents=True)
    obs, _ = env.reset()
    obs_dim = obs['observation'].shape[0]
    
    # Reti TD-Learning e Avversario (Self-Play)
    policy_net = ValueNet(obs_dim).to(device)
    target_net = ValueNet(obs_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    opponent_net = ValueNet(obs_dim).to(device)
    opponent_net.load_state_dict(policy_net.state_dict())
    opponent_net.eval()
    
    def opponent_policy(env_obj, player_id):
        return select_action(env_obj, opponent_net, epsilon=0.05, device=device, max_actions=10)
    
    env.opponent_policy_fn = opponent_policy
    
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=3e-4)
    buffer = NStepReplayBuffer(capacity=100_000, n_step=3, gamma=0.99)
    
    # Iperparametri
    batch_size = 128             # Più grande per sfruttare la GPU
    gamma = 0.99
    target_update = 2000
    learning_starts = 1000
    max_episode_steps = 500
    log_interval = 5000
    train_freq = 4               # Aggiorna ogni N step
    eval_freq = 50000            # Esegui valutazione greedy periodica
    
    eps_start = 1.0
    eps_end = 0.05
    eps_decay = int(0.3 * total_timesteps)
    
    state = obs['observation']
    episode_reward = 0
    episode_step = 0
    episodes = 0
    losses = []
    recent_rewards = deque(maxlen=50)
    steps_per_sec = deque(maxlen=100)
    best_avg_reward = -float('inf')
    
    print(f"Obs dim: {obs_dim} | Batch: {batch_size} | Device: {device}")
    print(f"Max ep steps: {max_episode_steps}")
    print("=" * 60)
    
    for step in range(total_timesteps):
        t0 = time.time()
        
        if step < 10:
            print(f"[DEBUG] Step {step} Start. Env phase: {env.game_phase}, CPU: {env.current_player}")
            
        # Epsilon decay lineare
        epsilon = eps_end + (eps_start - eps_end) * max(0, (eps_decay - step) / eps_decay)
        
        # 1. Seleziona azione (con save/restore, non deepcopy)
        if step < 10:
            print(f"[DEBUG] Step {step} Selecting action...")
        action_tuple = select_action(env, policy_net, epsilon, device)
        action_array = np.array(action_tuple)
        
        if step < 10:
            print(f"[DEBUG] Step {step} Action selected: {action_tuple}. Applying env.step...")
            
        # 2. Applica nel vero ambiente
        next_obs, reward, terminated, truncated, _ = env.step(action_array)
        next_state = next_obs['observation']
        episode_step += 1
        
        if step < 10:
            print(f"[DEBUG] Step {step} env.step finished! Term: {terminated}")
            
        if episode_step >= max_episode_steps:
            truncated = True
        done = terminated or truncated
        
        # 3. Salva transizione
        buffer.push(state, reward, next_state, done)
        state = next_state
        episode_reward += reward
        
        # 4. Ottimizzazione su GPU
        if step > learning_starts and step % train_freq == 0:
            s_batch, r_batch, s_next_batch, done_batch = buffer.sample(batch_size)
            
            s_batch = torch.FloatTensor(s_batch).to(device)
            r_batch = torch.FloatTensor(r_batch).to(device)
            s_next_batch = torch.FloatTensor(s_next_batch).to(device)
            done_batch = torch.FloatTensor(done_batch).to(device)
            
            with torch.no_grad():
                v_next = target_net(s_next_batch).squeeze(-1)
                n_step_gamma = gamma ** buffer.n_step
                target_v = r_batch + n_step_gamma * v_next * (1 - done_batch)
                
            v_curr = policy_net(s_batch).squeeze(-1)
            loss = F.mse_loss(v_curr, target_v)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()
            
            losses.append(loss.item())
        
        # 5. Update Target Network
        if step % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # FPS tracking
        step_time = time.time() - t0
        if step_time > 0:
            steps_per_sec.append(1.0 / step_time)
        
        # Fine episodio
        if done:
            obs, _ = env.reset()
            state = obs['observation']
            episodes += 1
            recent_rewards.append(episode_reward)
            
            if episodes % 5 == 0:
                avg_fps = np.mean(steps_per_sec) if steps_per_sec else 0
                avg_loss = np.mean(losses[-50:]) if losses else 0
                trunc_str = " [TRUNC]" if truncated and not terminated else ""
                print(f"[{step:>7}] Ep {episodes:>4} | R: {episode_reward:>7.1f} | "
                      f"Steps: {episode_step:>3} | Eps: {epsilon:.2f} | "
                      f"FPS: {avg_fps:>5.0f} | Loss: {avg_loss:.4f}{trunc_str}")
            
            episode_reward = 0
            episode_step = 0
        
        # Log periodico
        if step > 0 and step % log_interval == 0:
            avg_fps = np.mean(steps_per_sec) if steps_per_sec else 0
            avg_loss = np.mean(losses[-50:]) if losses else 0
            avg_r = np.mean(recent_rewards) if recent_rewards else 0
            print(f"--- Step {step:>7}/{total_timesteps} | Ep: {episodes} | "
                  f"AvgR(50): {avg_r:>7.1f} | Eps: {epsilon:.2f} | "
                  f"FPS: {avg_fps:>5.0f} | Loss: {avg_loss:.4f} ---")
        
        # Checkpoint
        if step > 0 and step % 50000 == 0:
            model_path = os.path.join(save_dir, f"avn_pinnacola_steps_{step}.pth")
            torch.save(policy_net.state_dict(), model_path)
            
        # Valutazione Periodica Greedy
        if step > 0 and step % eval_freq == 0:
            print(f"\n{'='*40}")
            print(f" 🔍 Esecuzione Validazione Greedy (Step {step})...")
            policy_net.eval()
            win_rate, avg_val_reward = evaluate_current_model(policy_net, device, n_episodes=20)
            policy_net.train()
            print(f" 📊 Risultati: Win Rate {win_rate:.1f}% | Avg Reward {avg_val_reward:.1f}")
            
            if avg_val_reward > best_avg_reward:
                best_avg_reward = avg_val_reward
                best_path = os.path.join(save_dir, "avn_pinnacola_best.pth")
                torch.save(policy_net.state_dict(), best_path)
                
                # Aggiorna il modello avversario per il Self-Play!
                opponent_net.load_state_dict(policy_net.state_dict())
                
                print(f" 🌟 Nuovo BEST model salvato in {best_path}!")
                print(" 🤖 Modello avversario aggiornato per il Self-Play!")
            print(f"{'='*40}\n")
    
    final_path = os.path.join(save_dir, "avn_pinnacola_final.pth")
    torch.save(policy_net.state_dict(), final_path)
    print(f"\n✅ Training completato! Modello salvato: {final_path}")
    return policy_net


def evaluate_avn(model_path, n_episodes=5):
    """Valutazione greedy del modello."""
    device = get_device()
    env = PinnacolaEnv()
    obs, _ = env.reset()
    
    model = ValueNet(obs['observation'].shape[0]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    wins = 0
    print(f"Evaluating: {model_path}")
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done, ep_reward, ep_steps = False, 0, 0
        
        while not done and ep_steps < 500:
            action_tuple = select_action(env, model, epsilon=0.0, device=device)
            obs, reward, terminated, truncated, _ = env.step(np.array(action_tuple))
            ep_reward += reward
            ep_steps += 1
            done = terminated or truncated
            
        is_win = ep_reward > 50
        wins += int(is_win)
        print(f"  Episode {ep+1}: Reward={ep_reward:.2f}, Steps={ep_steps}, Win={is_win}")
        
    print(f"\nWin Rate: {wins}/{n_episodes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", type=str)
    parser.add_argument("--timesteps", type=int, default=100_000)
    args = parser.parse_args()
    
    if args.train:
        train_avn(total_timesteps=args.timesteps)
    elif args.eval:
        evaluate_avn(args.eval)
