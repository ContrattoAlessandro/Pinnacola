"""
FASE 3-4: Backend FastAPI + Pipeline di Inferenza
Gestisce stato partita e interfacciamento con modello RL
"""

import os
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np

# Environment
import torch
from custom_avn_train import ValueNet, select_action, get_device
from pinnacola_env import PinnacolaEnv, ActionType, Card, CardRank, CardSuit

RL_AVAILABLE = True
device = get_device()

app = FastAPI(title="Pinnacola RL Assistant API")

# ==================== DATA MODELS ====================

class CardInput(BaseModel):
    rank: Optional[str] = None  # "A", "2", ... "K"
    suit: Optional[int] = None  # 0=♥, 1=♦, 2=♣, 3=♠
    deck: int = 0
    type: str = "normal"  # "normal" o "joker"

class StartGameRequest(BaseModel):
    hand: List[CardInput]
    num_players: int = 4

class PredictMoveRequest(BaseModel):
    game_id: str
    my_hand: List[CardInput]
    table_melds: List[Dict[str, Any]]
    discard_pile: List[CardInput]
    opponent_cards: List[int]
    phase: str  # "draw", "meld", "discard"

class MoveSuggestion(BaseModel):
    steps: List[str]
    summary: str
    action_code: List[int]  # [action_type, card_idx, meld_idx, param]
    confidence: float

class GameState(BaseModel):
    game_id: str
    created_at: str
    env_state: Optional[Dict] = None
    hand: List[CardInput]
    table_melds: List[Dict]
    discard_pile: List[CardInput]
    opponent_cards: List[int]
    current_phase: str


# ==================== STATE MANAGEMENT ====================

@dataclass
class ActiveGame:
    game_id: str
    env: PinnacolaEnv
    created_at: datetime = field(default_factory=datetime.now)
    last_action: Optional[str] = None

# In-memory storage (in produzione usare Redis/DB)
active_games: Dict[str, ActiveGame] = {}

# Modello RL (caricato lazy)
rl_model = None

def get_rl_model():
    """Lazy loading del modello RL."""
    global rl_model
    if rl_model is None and RL_AVAILABLE:
        model_path = os.getenv("MODEL_PATH", "./models/avn_pinnacola_best.pth")
        if os.path.exists(model_path):
            try:
                # Need to determine obs_dim to initialize custom ValueNet
                dummy_env = PinnacolaEnv()
                obs, _ = dummy_env.reset()
                obs_dim = obs['observation'].shape[0]
                
                model = ValueNet(obs_dim).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                model.eval()
                rl_model = model
                print(f"✅ Model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Failed to load model: {e}")
                rl_model = None
    return rl_model

# ==================== UTILS ====================

def card_input_to_env(card: CardInput, all_cards: List[Card]) -> Optional[Card]:
    """Converte CardInput in Card dell'environment."""
    if card.type == "joker":
        # Trova Jolly corrispondente
        for c in all_cards:
            if c.rank == CardRank.JOKER and c.deck_id == card.deck:
                return c
    else:
        rank_map = {'A': 1, 'J': 11, 'Q': 12, 'K': 13}
        rank_val = rank_map.get(card.rank, int(card.rank) if card.rank.isdigit() else 1)
        
        for c in all_cards:
            if (c.rank.value == rank_val and 
                c.suit.value == card.suit and 
                c.deck_id == card.deck):
                return c
    return None

def env_card_to_dict(card: Card) -> Dict:
    """Converte Card in dizionario per frontend."""
    if card.rank == CardRank.JOKER:
        return {"type": "joker", "deck": card.deck_id}
    
    rank_names = {1: 'A', 11: 'J', 12: 'Q', 13: 'K'}
    rank_str = rank_names.get(card.rank.value, str(card.rank.value))
    
    return {
        "rank": rank_str,
        "suit": card.suit.value,
        "deck": card.deck_id,
        "type": "normal"
    }

def action_to_italian(action_type: int, card: Optional[Card] = None, 
                       meld_idx: int = 0) -> str:
    """Converte azione in istruzione italiana."""
    card_str = str(card) if card else ""
    
    actions = {
        ActionType.DRAW_STOCK: "📥 Pesca dal tallone",
        ActionType.DRAW_PILE: "🗑️ Raccogli TUTTO il pozzo",
        ActionType.MELD_SET: f"🎴 Cala Tris con {card_str}",
        ActionType.MELD_RUN: f"🎴 Cala Scala con {card_str}",
        ActionType.ATTACH_CARD: f"🔗 Attacca {card_str} al gioco #{meld_idx}",
        ActionType.REPLACE_JOKER: f"🃏 Sostituisci Jolly con {card_str}",
        ActionType.SKIP_MELD: "⏭️ Non calare nulla ora",
        ActionType.DISCARD: f"🚮 Scarta {card_str}",
        ActionType.CLOSE_ROUND: "🏆 CHIUDI la partita!",
    }
    
    return actions.get(action_type, "Azione sconosciuta")

# ==================== API ENDPOINTS ====================

@app.post("/api/start_game", response_model=Dict[str, str])
async def start_game(request: StartGameRequest):
    """Inizializza una nuova partita con le carte del giocatore."""
    
    game_id = str(uuid.uuid4())[:8]
    
    # Crea environment
    env = PinnacolaEnv(num_players=request.num_players)
    obs, info = env.reset()
    
    # Sovrascrivi mano del bot con le carte fornite
    # Nota: qui facciamo un hack - in produzione bisognerebbe
    # reinizializzare l'env con le carte specifiche
    
    game = ActiveGame(game_id=game_id, env=env)
    active_games[game_id] = game
    
    return {
        "game_id": game_id,
        "status": "started",
        "message": "Partita iniziata. Tocca a te!"
    }

@app.post("/api/predict_move", response_model=MoveSuggestion)
async def predict_move(request: PredictMoveRequest):
    """
    Interroga il modello RL per ottenere il suggerimento migliore.
    Se il modello non è disponibile, usa regole euristiche.
    """
    
    game = active_games.get(request.game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    env = game.env
    
    # Ottieni observation corrente
    obs = env._get_observation()
    mask = env._get_action_mask()
    
    # Cerca modello RL
    model = get_rl_model()
    
    if model and RL_AVAILABLE:
        # Usa modello RL custom AVN
        action_tuple = select_action(env, model, epsilon=0.0, device=device, max_actions=50)
        action_type = int(action_tuple[0])
        card_idx = int(action_tuple[1])
        meld_idx = int(action_tuple[2])
        param = int(action_tuple[3])
        confidence = 0.9
    else:
        # Fallback: regole euristiche
        action_type, card_idx, meld_idx, param, confidence = heuristic_move(env, mask)
    
    # Converti in istruzioni italiane
    card = env.idx_to_card.get(card_idx)
    
    # Costruisci lista di step
    steps = []
    
    if env.game_phase == 0:  # DRAW
        if action_type == ActionType.DRAW_PILE and env.discard_pile:
            steps.append("🗑️ Raccogli TUTTE le carte dal pozzo")
            steps.append("⚠️ Attenzione: prendi anche le carte che non ti servono!")
        else:
            steps.append("📥 Pesca dal tallone (carta coperta)")
        steps.append("➡️ Passa alla fase calata")
    
    elif env.game_phase == 1:  # MELD
        if action_type == ActionType.MELD_SET or action_type == ActionType.MELD_RUN:
            meld_type = "Tris" if action_type == ActionType.MELD_SET else "Scala"
            steps.append(f"🎴 Cala un {meld_type}: {card}")
            # Trova carte complete del meld
            valid_melds = env._find_valid_melds(env.player_hands[0])
            if valid_melds:
                meld_cards = valid_melds[0]  # Semplificazione
                cards_str = " - ".join(str(c) for c in meld_cards)
                steps.append(f"   Carte: {cards_str}")
        
        elif action_type == ActionType.ATTACH_CARD:
            meld = env.table_melds[meld_idx] if meld_idx < len(env.table_melds) else None
            meld_desc = str(meld) if meld else f"gioco #{meld_idx}"
            steps.append(f"🔗 Attacca {card} a: {meld_desc}")
        
        elif action_type == ActionType.REPLACE_JOKER:
            steps.append(f"🃏 Sostituisci il Jolly con {card}")
            steps.append("   (Ritira il Jolly in mano)")
        
        else:  # SKIP_MELD
            steps.append("⏭️ Non calare nulla in questa fase")
        
        steps.append("➡️ Passa alla fase scarto")
    
    elif env.game_phase == 2:  # DISCARD
        if action_type == ActionType.CLOSE_ROUND:
            steps.append("🏆 CHIUDI la partita!")
            steps.append("   Hai calato tutto e non hai carte in mano")
        else:
            steps.append(f"🚮 Scarta sul pozzo: {card}")
            steps.append("   Fine del tuo turno")
    
    # Summary
    summary = action_to_italian(action_type, card, meld_idx)
    
    return MoveSuggestion(
        steps=steps,
        summary=summary,
        action_code=[action_type, card_idx, meld_idx, param],
        confidence=confidence
    )

@app.post("/api/execute_action")
async def execute_action(game_id: str, action: List[int] = Body(...)):
    """Esegue un'azione nel gioco e aggiorna lo stato."""
    
    game = active_games.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    env = game.env
    action_array = np.array(action)
    
    obs, reward, terminated, truncated, info = env.step(action_array)
    
    # Se il round finisce, simula turno avversario
    if env.current_player != 0 and not terminated:
        env._simulate_opponent_turns()
    
    return {
        "reward": float(reward),
        "terminated": terminated,
        "truncated": truncated,
        "phase": env.game_phase.name,
        "my_cards": len(env.player_hands[0]),
        "opponent_cards": [len(env.player_hands[i]) for i in range(1, env.num_players)]
    }

@app.get("/api/game_state/{game_id}")
async def get_game_state(game_id: str):
    """Restituisce lo stato completo del gioco."""
    
    game = active_games.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    env = game.env
    
    return {
        "game_id": game_id,
        "phase": env.game_phase.name,
        "turn": env.turn_count,
        "my_hand": [env_card_to_dict(c) for c in env.player_hands[0]],
        "table_melds": [
            {
                "id": m.meld_id,
                "type": m.meld_type,
                "cards": [env_card_to_dict(c) for c in m.cards],
                "owner": m.owner
            }
            for m in env.table_melds
        ],
        "discard_pile": [env_card_to_dict(c) for c in env.discard_pile],
        "opponent_cards": [len(env.player_hands[i]) for i in range(1, env.num_players)],
        "stock_remaining": len(env.stock_pile)
    }

# ==================== HEURISTIC FALLBACK ====================

def heuristic_move(env: PinnacolaEnv, action_mask: np.ndarray) -> tuple:
    """
    Fallback euristico quando il modello RL non è disponibile.
    Implementa regole base per giocare decentemente.
    """
    legal_actions = env._get_legal_actions()
    
    if not legal_actions:
        return (0, 0, 0, 0, 0.0)  # Azione di default
    
    hand = env.player_hands[0]
    phase = env.game_phase
    
    # Priorità fase DRAW
    if phase == 0:
        # Preferisci pescare dal tallone (meno informazioni date agli avversari)
        for a in legal_actions:
            if a[0] == ActionType.DRAW_STOCK:
                return (*a, 0.6)
        # Altrimenti prendi dal pozzo
        return (*legal_actions[0], 0.5)
    
    # Priorità fase MELD
    elif phase == 1:
        # Priorità: sostituisci Jolly > cala combinazione > attacca > skip
        for a in legal_actions:
            if a[0] == ActionType.REPLACE_JOKER:
                return (*a, 0.85)
        
        for a in legal_actions:
            if a[0] in [ActionType.MELD_SET, ActionType.MELD_RUN]:
                return (*a, 0.8)
        
        for a in legal_actions:
            if a[0] == ActionType.ATTACH_CARD:
                return (*a, 0.7)
        
        # Skip se non c'è nient'altro
        for a in legal_actions:
            if a[0] == ActionType.SKIP_MELD:
                return (*a, 0.3)
    
    # Priorità fase DISCARD
    elif phase == 2:
        # Prova a chiudere se possibile
        for a in legal_actions:
            if a[0] == ActionType.CLOSE_ROUND:
                return (*a, 1.0)
        
        # Altrimenti scarta carta più alta (semplificazione)
        for a in legal_actions:
            if a[0] == ActionType.DISCARD:
                card = env.idx_to_card.get(a[1])
                if card and card.rank != CardRank.JOKER:
                    # Preferisci scartare K, Q, J
                    if card.rank.value >= 11:
                        return (*a, 0.7)
        
        # Scarta qualsiasi carta
        for a in legal_actions:
            if a[0] == ActionType.DISCARD:
                return (*a, 0.5)
    
    return (*legal_actions[0], 0.5)

# ==================== STATIC FILES ====================

# Serve frontend static files
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("🚀 Pinnacola RL Assistant API")
    print("=" * 50)
    
    # Check modello
    model = get_rl_model()
    if model:
        print("✅ RL Model: Loaded")
    else:
        print("⚠️  RL Model: Not available (using heuristics)")
    
    print("\n📱 Frontend: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
