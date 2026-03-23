# Pinnacola RL Assistant

Sistema di AI addestrata con Reinforcement Learning per giocare a Pinnacola e assistere durante partite reali.

## 🎯 Obiettivo

Costruire un bot Pinnacola che:
- Gioca in simulazione tramite RL (PPO)
- Funge da "Assistente dal vivo" durante partite reali
- Riceve input da smartphone e suggerisce mosse ottimali

## 📁 Struttura Progetto

```
Pinnacola/
├── requirements.txt          # Dipendenze Python
├── pinnacola_env.py          # Environment Gymnasium (FASE 1)
├── train.py                  # Training PPO (FASE 2)
├── test_env.py               # Test environment
├── main.py                   # Backend FastAPI (FASE 3-4)
└── frontend/
    └── index.html            # Web app mobile (FASE 3)
```

## 🚀 Setup

```bash
# Installa dipendenze
pip install -r requirements.txt

# Test environment
python test_env.py
```

## 🤖 Training (FASE 2)

```bash
# Training base (500k steps)
python train.py --train --timesteps 500000

# Training completo (1M+ steps)  
python train.py --train --timesteps 1000000

# Valuta modello
python train.py --eval models/ppo_pinnacola_final.zip --episodes 20
```

### Metriche Training

Il training monitora automaticamente:
- **Win Rate**: % partite vinte
- **Avg Reward**: Reward medio episodio  
- **Avg Length**: Lunghezza media partite

Log salvati in `models/training_metrics.csv`.

## 📱 Frontend (FASE 3)

```bash
# Avvia backend + frontend
python main.py
# Apri http://localhost:8000 sullo smartphone
```

### Features Mobile
- **Setup veloce**: Griglia 7xN per selezione 13 carte
- **Input avversario**: Pulsanti rapidi "Ha pescato/Ha calato/Ha scartato"
- **Assistente AI**: "COSA FACCIO?" con istruzioni in italiano
- **One-hand use**: Touch-optimized, minimal clicks

## 🖥️ Backend (FASE 4)

### API Endpoints
- `POST /api/start_game` - Inizializza partita
- `POST /api/predict_move` - Chiede mossa all'AI
- `POST /api/execute_action` - Esegue azione
- `GET /api/game_state/{id}` - Stato completo

### Fallback
Se il modello RL non è disponibile, usa regole euristiche base.

## 🎮 Come Usare

1. **Training**: Addestra il modello con PPO
2. **Avvio Backend**: `python main.py`
3. **Setup Partita**: Inserisci 13 carte iniziali
4. **Input Avversario**: Registra mosse avversari
5. **AI Assist**: Premi "COSA FACCIO?" per suggerimenti

## 📊 Performance

| Metrica | Target | Buono | Ottimo |
|---------|--------|-------|--------|
| WinRate | >20% | >40% | >60% |
| AvgReward | -50 | -10 | >+50 |
| AvgLength | 150 | 100 | <80 |

## 🔧 Stack Tecnologico

- **RL**: Stable Baselines3 + PPO
- **Backend**: FastAPI + Python
- **Frontend**: HTML/JS + TailwindCSS
- **Environment**: Gymnasium
- **Training**: Multi-process parallel environments

## 🤝 Contributi

1. Fork repository
2. Crea feature branch
3. Commit changes
4. Push to branch
5. Open Pull Request

## 📄 Licenza

MIT License - vedi LICENSE file

---

**Status**: ✅ Fase 1-4 completate | 🚧 Training in corso
