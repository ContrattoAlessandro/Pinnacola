"""
Pinnacola RL Assistant - Environment Module
FASE 1: Environment Gymnasium per Pinnacola
"""

import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from enum import IntEnum
from dataclasses import dataclass, field


class CardRank(IntEnum):
    """Rappresentazione dei valori delle carte (2 mazzi francesi)."""
    ACE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    JOKER = 14  # Jolly


class CardSuit(IntEnum):
    """Semi delle carte."""
    HEARTS = 0    # Cuori
    DIAMONDS = 1  # Quadri
    CLUBS = 2     # Fiori
    SPADES = 3    # Picche
    NONE = 4      # Per Jolly (senza seme)


@dataclass(frozen=True)
class Card:
    """Rappresentazione immutabile di una carta."""
    rank: CardRank
    suit: CardSuit
    deck_id: int  # 0 o 1 per distinguere i due mazzi (evita carte identiche)
    
    def __hash__(self):
        return hash((self.rank, self.suit, self.deck_id))
    
    def __repr__(self):
        if self.rank == CardRank.JOKER:
            return f"JOKER{self.deck_id}"
        rank_names = {1: 'A', 11: 'J', 12: 'Q', 13: 'K'}
        rank_str = rank_names.get(self.rank, str(self.rank))
        suit_symbols = ['♥', '♦', '♣', '♠', '']
        return f"{rank_str}{suit_symbols[self.suit]}{self.deck_id}"


class GamePhase(IntEnum):
    """Fasi di un turno in Pinnacola."""
    DRAW = 0      # Pesca dal tallone o dal pozzo
    MELD = 1      # Cala combinazioni o attacca carte
    DISCARD = 2   # Scarta per terminare il turno
    ROUND_OVER = 3  # Round terminato


class ActionType(IntEnum):
    """Tipi di azione possibili."""
    # Fase DRAW
    DRAW_STOCK = 0      # Pesca dal tallone
    DRAW_PILE = 1       # Raccogli dal pozzo (con tutte le carte)
    
    # Fase MELD
    MELD_SET = 2        # Cala un tris/quartetto
    MELD_RUN = 3        # Cala una scala
    ATTACH_CARD = 4     # Attacca carta a gioco esistente
    REPLACE_JOKER = 5   # Sostituisci un Jolly
    SKIP_MELD = 6       # Salta fase calata
    
    # Fase DISCARD
    DISCARD = 7         # Scarta una carta
    CLOSE_ROUND = 8     # Chiudi il round (se possibile)


@dataclass
class Meld:
    """Rappresenta un gioco calato sul tavolo."""
    meld_id: int
    meld_type: str  # 'set' (tris) o 'run' (scala)
    cards: List[Card] = field(default_factory=list)
    owner: int = 0  # 0 = bot, 1+ = avversari
    
    def can_attach(self, card: Card) -> bool:
        """Verifica se una carta può essere attaccata a questo gioco."""
        if self.meld_type == 'set':
            # Tris: stesso rank, qualsiasi seme
            return card.rank == self.cards[0].rank
        elif self.meld_type == 'run':
            # Scala: stesso seme, consecutiva
            ranks = sorted([c.rank for c in self.cards if c.rank != CardRank.JOKER])
            if not ranks:
                return False
            suit = self.cards[0].suit if self.cards[0].rank != CardRank.JOKER else self.cards[1].suit
            if card.suit != suit:
                return False
            # Può estendere in alto o in basso
            return card.rank == ranks[0] - 1 or card.rank == ranks[-1] + 1
        return False
    
    def has_joker(self) -> bool:
        """Verifica se il gioco contiene un Jolly sostituibile."""
        return any(c.rank == CardRank.JOKER for c in self.cards)
    
    def get_replaceable_joker(self) -> Optional[Card]:
        """Restituisce il Jolly sostituibile se presente."""
        for card in self.cards:
            if card.rank == CardRank.JOKER:
                return card
        return None


class PinnacolaEnv(gym.Env):
    """
    Environment Gymnasium per Pinnacola (2 mazzi, 4 giocatori).
    
    Observation Space (stato piatto):
    - Mano del bot: 108-dim one-hot (13 carte max, padding se < 13)
    - Pozzo (top): 108-dim one-hot (cima del discard pile)
    - Carte nel pozzo (tutte): 108-dim count (quante copie di ogni carta)
    - Giochi sul tavolo: lista di melds codificati
    - Carte in mano agli avversari: 3 valori (n carte ciascuno)
    - Carte viste (memoria): 108-dim count (carte già passate/uscite)
    - Fase del turno: 4-dim one-hot
    
    Action Space (fattorizzato con masking):
    - Tipo azione: 9 valori (ActionType)
    - Carta target: 108 (indice della carta specifica)
    - Meld target: 20 max (indice del gioco su cui operare)
    - Parametro extra: 10-dim (per sostituzioni, estensioni, ecc.)
    
    Action Masking:
    Il metodo _get_action_mask() genera una maschera booleana che indica
    quali azioni sono legali nello stato corrente. Questo è fondamentale
    perché il 99% delle combinazioni non è valida in un dato turno.
    """
    
    # Costanti del gioco
    NUM_DECKS = 2
    CARDS_PER_DECK = 54  # 52 + 2 Jolly
    TOTAL_CARDS = NUM_DECKS * CARDS_PER_DECK  # 108
    NUM_PLAYERS = 4
    CARDS_PER_PLAYER = 13
    MAX_MELDS = 20  # Massimo numero di giochi sul tavolo
    MAX_HAND_SIZE = 20  # Dimensione max mano (con pozzo raccolto)
    
    # Punteggi
    POINTS_MELD_SET = 5
    POINTS_MELD_RUN = 10
    POINTS_ATTACH = 2
    POINTS_REPLACE_JOKER = 15
    PENALTY_CARD_IN_HAND = 5  # Punti persi per carta rimasta
    REWARD_WIN = 100
    REWARD_LOSE = -100
    
    def __init__(self, num_players: int = 4):
        super().__init__()
        
        self.num_players = num_players
        self.bot_player_id = 0  # Il bot è sempre il giocatore 0
        
        # Inizializza i due mazzi di carte (108 carte totali)
        self.all_cards = self._create_decks()
        self.card_to_idx = {card: i for i, card in enumerate(self.all_cards)}
        self.idx_to_card = {i: card for i, card in enumerate(self.all_cards)}
        
        # Observation Space
        # Calcola dimensione totale dello stato piatto
        obs_dim = (
            self.TOTAL_CARDS +  # Mano bot (one-hot multiplo, max 13 carte)
            self.TOTAL_CARDS +  # Pozzo top
            self.TOTAL_CARDS +  # Carte nel pozzo (count)
            self.MAX_MELDS * (self.TOTAL_CARDS + 3) +  # Giochi: carte + metadata
            (self.num_players - 1) +  # Carte avversari
            self.TOTAL_CARDS +  # Carte viste (memoria)
            4  # Fase turno
        )
        
        self.observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Box(
                low=0, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
            'action_mask': gym.spaces.Box(
                low=0, high=1, shape=(self._get_action_space_size(),), dtype=np.int8
            )
        })
        
        # Action Space: (action_type, card_idx, meld_idx, param)
        # Usiamo MultiDiscrete per rappresentare azioni fattorizzate
        self.action_space = gym.spaces.MultiDiscrete([
            len(ActionType),  # Tipo azione
            self.TOTAL_CARDS,  # Carta specifica
            self.MAX_MELDS,    # Meld target
            10                 # Parametro extra
        ])
        
        # Stato del gioco (inizializzato in reset)
        self.player_hands: List[List[Card]] = []
        self.stock_pile: List[Card] = []  # Tallone
        self.discard_pile: List[Card] = []  # Pozzo
        self.table_melds: List[Meld] = []
        self.cards_seen: np.ndarray = None  # Memoria carte uscite
        self.current_player: int = 0
        self.game_phase: GamePhase = GamePhase.DRAW
        self.round_over: bool = False
        self.turn_count: int = 0
        
        # Stato specifico turno bot
        self.bot_has_drawn: bool = False
        self.bot_melded_this_turn: bool = False
        self.can_close_this_turn: bool = False
        
    def _create_decks(self) -> List[Card]:
        """Crea i due mazzi da 54 carte."""
        cards = []
        for deck_id in range(self.NUM_DECKS):
            # Carte standard 1-10, J, Q, K
            for rank in range(1, 14):
                for suit in range(4):
                    cards.append(Card(
                        rank=CardRank(rank),
                        suit=CardSuit(suit),
                        deck_id=deck_id
                    ))
            # 2 Jolly per mazzo
            cards.append(Card(rank=CardRank.JOKER, suit=CardSuit.NONE, deck_id=deck_id))
            cards.append(Card(rank=CardRank.JOKER, suit=CardSuit.NONE, deck_id=deck_id))
        return cards
    
    def _get_action_space_size(self) -> int:
        """Calcola dimensione flat dell'action space per masking."""
        # Semplificazione: action_type * card * meld * param
        # In pratica useremo un encoding più smart, ma per la maschera
        # ci serve una dimensione fissa
        return len(ActionType) * self.TOTAL_CARDS * self.MAX_MELDS * 10
    
    def _encode_hand(self, hand: List[Card]) -> np.ndarray:
        """Codifica mano come vettore one-hot count (108-dim)."""
        encoded = np.zeros(self.TOTAL_CARDS, dtype=np.float32)
        for card in hand:
            idx = self.card_to_idx[card]
            encoded[idx] += 1
        return encoded
    
    def _encode_melds(self) -> np.ndarray:
        """Codifica giochi sul tavolo."""
        # Ogni meld: one-hot carte + tipo + owner
        encoded = np.zeros(self.MAX_MELDS * (self.TOTAL_CARDS + 3), dtype=np.float32)
        for i, meld in enumerate(self.table_melds[:self.MAX_MELDS]):
            base_idx = i * (self.TOTAL_CARDS + 3)
            # Carte nel meld
            for card in meld.cards:
                card_idx = self.card_to_idx[card]
                encoded[base_idx + card_idx] = 1
            # Metadata: tipo (0=set, 1=run)
            encoded[base_idx + self.TOTAL_CARDS] = 0 if meld.meld_type == 'set' else 1
            # Owner
            encoded[base_idx + self.TOTAL_CARDS + 1] = meld.owner
            # Numero carte
            encoded[base_idx + self.TOTAL_CARDS + 2] = len(meld.cards)
        return encoded
    
    def _get_observation(self) -> np.ndarray:
        """Compone l'observation vector completo."""
        obs_parts = []
        
        # 1. Mano del bot (one-hot)
        bot_hand = self.player_hands[self.bot_player_id]
        obs_parts.append(self._encode_hand(bot_hand))
        
        # 2. Carta in cima al pozzo
        top_discard = np.zeros(self.TOTAL_CARDS, dtype=np.float32)
        if self.discard_pile:
            top_idx = self.card_to_idx[self.discard_pile[-1]]
            top_discard[top_idx] = 1
        obs_parts.append(top_discard)
        
        # 3. Tutte le carte nel pozzo (count)
        discard_count = np.zeros(self.TOTAL_CARDS, dtype=np.float32)
        for card in self.discard_pile:
            discard_count[self.card_to_idx[card]] += 1
        obs_parts.append(discard_count)
        
        # 4. Giochi sul tavolo
        obs_parts.append(self._encode_melds())
        
        # 5. Numero carte in mano agli avversari
        opponent_cards = np.zeros(self.num_players - 1, dtype=np.float32)
        for i in range(1, self.num_players):
            opponent_cards[i-1] = len(self.player_hands[i]) / self.MAX_HAND_SIZE
        obs_parts.append(opponent_cards)
        
        # 6. Carte viste (memoria per probabilità)
        obs_parts.append(self.cards_seen.astype(np.float32) / self.NUM_DECKS)
        
        # 7. Fase del turno (one-hot)
        phase = np.zeros(4, dtype=np.float32)
        phase[self.game_phase] = 1
        obs_parts.append(phase)
        
        return np.concatenate(obs_parts)
    
    def _get_legal_actions(self) -> List[Tuple[int, int, int, int]]:
        """
        Restituisce lista di azioni legali come tuple.
        Usato per generare la action mask.
        """
        legal = []
        bot_hand = self.player_hands[self.bot_player_id]
        
        if self.game_phase == GamePhase.DRAW:
            # Può pescare dal tallone se disponibile
            if self.stock_pile:
                for card in bot_hand:
                    legal.append((ActionType.DRAW_STOCK, self.card_to_idx[card], 0, 0))
            # Può raccogliere dal pozzo
            if self.discard_pile:
                for card in bot_hand:
                    legal.append((ActionType.DRAW_PILE, self.card_to_idx[card], 0, 0))
                    
        elif self.game_phase == GamePhase.MELD:
            # Può calare combinazioni
            melds_possible = self._find_valid_melds(bot_hand)
            for meld_cards in melds_possible:
                for card in meld_cards:
                    legal.append((ActionType.MELD_SET, self.card_to_idx[card], 0, len(meld_cards)))
            
            # Può attaccare carte ai giochi esistenti
            for meld_idx, meld in enumerate(self.table_melds):
                for card in bot_hand:
                    if meld.can_attach(card):
                        legal.append((ActionType.ATTACH_CARD, self.card_to_idx[card], meld_idx, 0))
            
            # Può sostituire Jolly
            for meld_idx, meld in enumerate(self.table_melds):
                if meld.has_joker() and meld.owner == self.bot_player_id:
                    joker = meld.get_replaceable_joker()
                    for card in bot_hand:
                        # Logica di sostituzione
                        if self._can_replace_joker(meld, card):
                            legal.append((ActionType.REPLACE_JOKER, self.card_to_idx[card], meld_idx, 0))
            
            # Può saltare la fase
            legal.append((ActionType.SKIP_MELD, 0, 0, 0))
            
        elif self.game_phase == GamePhase.DISCARD:
            # Deve scartare una carta dalla mano
            for card in bot_hand:
                legal.append((ActionType.DISCARD, self.card_to_idx[card], 0, 0))
            # Può chiudere se ha calato e mano vuota
            if self.bot_melded_this_turn and len(bot_hand) == 0:
                legal.append((ActionType.CLOSE_ROUND, 0, 0, 0))
        
        return legal
    
    def _find_valid_melds(self, hand: List[Card]) -> List[List[Card]]:
        """Trova tutte le combinazioni valide (tris/scale) nella mano."""
        valid_melds = []
        
        # Raggruppa carte per rank (per tris/quartetti)
        rank_groups: Dict[int, List[Card]] = {}
        for card in hand:
            if card.rank == CardRank.JOKER:
                continue
            rank = card.rank
            if rank not in rank_groups:
                rank_groups[rank] = []
            rank_groups[rank].append(card)
        
        # Conta Jolly
        jokers = [c for c in hand if c.rank == CardRank.JOKER]
        num_jokers = len(jokers)
        
        # Trova tris (3+ dello stesso rank, semi diversi)
        for rank, cards in rank_groups.items():
            if len(cards) >= 3:
                # Tris senza Jolly
                valid_melds.append(cards[:3])
                if len(cards) >= 4:
                    valid_melds.append(cards[:4])
            elif len(cards) == 2 and num_jokers >= 1:
                # Tris con Jolly
                valid_melds.append(cards + [jokers[0]])
        
        # Trova scale (4+ carte consecutive, stesso seme)
        for suit in range(4):  # Per ogni seme
            suit_cards = [c for c in hand if c.suit == CardSuit(suit) and c.rank != CardRank.JOKER]
            suit_cards.sort(key=lambda c: c.rank)
            
            # Trova sequenze consecutive
            if len(suit_cards) >= 4:
                # Cerca tutte le scale possibili (minimo 4 carte)
                for i in range(len(suit_cards)):
                    for j in range(i + 4, min(i + 14, len(suit_cards) + 1)):
                        subset = suit_cards[i:j]
                        # Verifica se sono consecutive
                        ranks = [c.rank for c in subset]
                        if len(ranks) >= 4 and max(ranks) - min(ranks) == len(ranks) - 1:
                            valid_melds.append(subset[:])
            
            # Scale con Jolly (es: 3-4-5 + Jolly = 3-4-5-6 oppure 2-3-4 + Jolly = A-2-3-4 o 2-3-4-5)
            if len(suit_cards) >= 3 and num_jokers >= 1:
                for i in range(len(suit_cards) - 2):
                    subset = suit_cards[i:i+3]
                    ranks = [c.rank for c in subset]
                    # Verifica gap di 1 (es: 3-4-6, Jolly può essere 5)
                    if ranks[1] == ranks[0] + 1 and ranks[2] == ranks[1] + 1:
                        # Scale consecutive con Jolly aggiunto
                        valid_melds.append(subset + [jokers[0]])
        
        return valid_melds
    
    def _can_replace_joker(self, meld: Meld, card: Card) -> bool:
        """Verifica se una carta può sostituire un Jolly nel meld."""
        if not meld.has_joker():
            return False
        
        # Trova il Jolly nel meld
        joker = None
        for c in meld.cards:
            if c.rank == CardRank.JOKER:
                joker = c
                break
        
        if not joker:
            return False
        
        if meld.meld_type == 'set':
            # In un tris, il Jolly rappresenta il rank del tris
            return card.rank == meld.cards[0].rank and card.rank != CardRank.JOKER
        elif meld.meld_type == 'run':
            # In una scala, il Jolly rappresenta un rank specifico nella sequenza
            # Trova che rank rappresenta il Jolly
            ranks = sorted([c.rank for c in meld.cards if c.rank != CardRank.JOKER])
            suits = [c.suit for c in meld.cards if c.suit != CardSuit.NONE]
            if not ranks or not suits:
                return False
            
            suit = suits[0]
            if card.suit != suit:
                return False
            
            # Il Jolly può essere in qualsiasi posizione della scala
            # Verifica se la carta potrebbe completare la sequenza
            possible_gaps = []
            for i in range(len(ranks) - 1):
                if ranks[i+1] - ranks[i] > 1:
                    possible_gaps.extend(range(ranks[i] + 1, ranks[i+1]))
            
            # O estendi oltre gli estremi
            possible_ranks = possible_gaps + [ranks[0] - 1, ranks[-1] + 1]
            return card.rank in possible_ranks
        
        return False
    
    def _get_action_mask(self) -> np.ndarray:
        """Genera maschera booleana per azioni legali."""
        mask = np.zeros(self._get_action_space_size(), dtype=np.int8)
        legal_actions = self._get_legal_actions()
        
        for action in legal_actions:
            # Flatten action tuple to index
            # Encoding: ((action_type * cards + card) * melds + meld) * params + param
            flat_idx = (
                action[0] * self.TOTAL_CARDS * self.MAX_MELDS * 10 +
                action[1] * self.MAX_MELDS * 10 +
                action[2] * 10 +
                action[3]
            )
            if flat_idx < len(mask):
                mask[flat_idx] = 1
        
        return mask
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Resetta l'ambiente per una nuova partita."""
        super().reset(seed=seed)
        
        # Mischia e distribuisci carte
        deck = self.all_cards.copy()
        self.np_random.shuffle(deck)
        
        self.player_hands = [[] for _ in range(self.num_players)]
        for player in range(self.num_players):
            for _ in range(self.CARDS_PER_PLAYER):
                if deck:
                    self.player_hands[player].append(deck.pop())
        
        # Tallone e pozzo
        self.stock_pile = deck[:-1]  # Tutto tranne ultima carta
        self.discard_pile = [deck[-1]] if deck else []  # Ultima carta scoperta
        
        # Reset stato
        self.table_melds = []
        self.cards_seen = np.zeros(self.TOTAL_CARDS, dtype=np.int32)
        
        # Registra carte già viste (carte in mano bot + pozzo)
        for card in self.player_hands[self.bot_player_id]:
            self.cards_seen[self.card_to_idx[card]] += 1
        for card in self.discard_pile:
            self.cards_seen[self.card_to_idx[card]] += 1
        
        self.current_player = 0
        self.game_phase = GamePhase.DRAW
        self.round_over = False
        self.turn_count = 0
        self.bot_has_drawn = False
        self.bot_melded_this_turn = False
        self.can_close_this_turn = False
        
        obs = self._get_observation()
        mask = self._get_action_mask()
        
        return {'observation': obs, 'action_mask': mask}, {}
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Esegue un'azione nel gioco.
        
        Args:
            action: np.ndarray [action_type, card_idx, meld_idx, param]
        
        Returns:
            observation: Dict con 'observation' e 'action_mask'
            reward: float
            terminated: bool
            truncated: bool
            info: Dict con info addizionali
        """
        action_type, card_idx, meld_idx, param = action
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        # Converti a int scalari
        action_type = int(action_type)
        card_idx = int(card_idx)
        meld_idx = int(meld_idx)
        param = int(param)
        
        # Esegui azione
        if action_type == ActionType.DRAW_STOCK:
            reward += self._action_draw_stock()
        elif action_type == ActionType.DRAW_PILE:
            reward += self._action_draw_pile()
        elif action_type == ActionType.MELD_SET or action_type == ActionType.MELD_RUN:
            reward += self._action_meld(action_type, card_idx, param)
        elif action_type == ActionType.ATTACH_CARD:
            reward += self._action_attach(card_idx, meld_idx)
        elif action_type == ActionType.REPLACE_JOKER:
            reward += self._action_replace_joker(card_idx, meld_idx)
        elif action_type == ActionType.SKIP_MELD:
            reward += self._action_skip_meld()
        elif action_type == ActionType.DISCARD:
            reward += self._action_discard(card_idx)
        elif action_type == ActionType.CLOSE_ROUND:
            reward += self._action_close_round()
        
        # Simula turni avversari se necessario
        if self.current_player != self.bot_player_id and not self.round_over:
            self._simulate_opponent_turns()
        
        # Verifica fine round
        if self.round_over:
            terminated = True
            # Calcola reward finale
            reward += self._calculate_final_reward()
        
        obs = self._get_observation()
        mask = self._get_action_mask()
        
        return {'observation': obs, 'action_mask': mask}, reward, terminated, truncated, info
    
    def _action_draw_stock(self) -> float:
        """Pesca dal tallone."""
        if not self.stock_pile or self.game_phase != GamePhase.DRAW:
            return -1.0  # Penalità azione illegale
        
        card = self.stock_pile.pop()
        self.player_hands[self.bot_player_id].append(card)
        self.cards_seen[self.card_to_idx[card]] += 1
        self.bot_has_drawn = True
        self.game_phase = GamePhase.MELD
        return 0.1  # Piccola reward per azione valida
    
    def _action_draw_pile(self) -> float:
        """Raccoglie tutto il pozzo."""
        if not self.discard_pile or self.game_phase != GamePhase.DRAW:
            return -1.0
        
        # Raccoglie tutte le carte dal pozzo
        cards_to_take = self.discard_pile[:]
        self.discard_pile = []
        self.player_hands[self.bot_player_id].extend(cards_to_take)
        
        for card in cards_to_take:
            self.cards_seen[self.card_to_idx[card]] += 1
        
        self.bot_has_drawn = True
        self.game_phase = GamePhase.MELD
        return 0.1
    
    def _action_meld(self, meld_type: int, card_idx: int, num_cards: int) -> float:
        """Cala una combinazione."""
        if self.game_phase != GamePhase.MELD:
            return -1.0
        
        # Trova la combinazione valida che corrisponde
        bot_hand = self.player_hands[self.bot_player_id]
        valid_melds = self._find_valid_melds(bot_hand)
        
        if not valid_melds:
            return -1.0  # Nessuna combinazione valida
        
        # Prendi la prima combinazione valida (semplificazione)
        # In un sistema completo, il parametro indicherebbe quale combinazione
        meld_cards = valid_melds[0]
        
        # Rimuovi carte dalla mano
        for card in meld_cards:
            if card in bot_hand:
                bot_hand.remove(card)
        
        # Determina tipo
        is_set = len(set(c.rank for c in meld_cards if c.rank != CardRank.JOKER)) == 1
        meld_type_str = 'set' if is_set else 'run'
        
        # Crea il meld
        new_meld = Meld(
            meld_id=len(self.table_melds),
            meld_type=meld_type_str,
            cards=meld_cards[:],
            owner=self.bot_player_id
        )
        self.table_melds.append(new_meld)
        self.bot_melded_this_turn = True
        
        # Reward basata sul tipo
        if meld_type_str == 'set':
            return self.POINTS_MELD_SET * len(meld_cards)
        else:
            return self.POINTS_MELD_RUN * len(meld_cards)
    
    def _action_attach(self, card_idx: int, meld_idx: int) -> float:
        """Attacca carta a gioco esistente."""
        if self.game_phase != GamePhase.MELD:
            return -1.0
        
        if meld_idx >= len(self.table_melds):
            return -1.0
        
        card = self.idx_to_card.get(card_idx)
        if not card or card not in self.player_hands[self.bot_player_id]:
            return -1.0
        
        meld = self.table_melds[meld_idx]
        
        if not meld.can_attach(card):
            return -1.0
        
        # Rimuovi carta dalla mano e aggiungi al meld
        self.player_hands[self.bot_player_id].remove(card)
        meld.cards.append(card)
        
        return self.POINTS_ATTACH
    
    def _action_replace_joker(self, card_idx: int, meld_idx: int) -> float:
        """Sostituisce un Jolly."""
        if self.game_phase != GamePhase.MELD:
            return -1.0
        
        if meld_idx >= len(self.table_melds):
            return -1.0
        
        card = self.idx_to_card.get(card_idx)
        if not card or card not in self.player_hands[self.bot_player_id]:
            return -1.0
        
        meld = self.table_melds[meld_idx]
        
        if meld.owner != self.bot_player_id:
            return -1.0
        
        if not self._can_replace_joker(meld, card):
            return -1.0
        
        # Trova e rimuovi il Jolly
        joker = meld.get_replaceable_joker()
        if not joker:
            return -1.0
        
        meld.cards.remove(joker)
        meld.cards.append(card)
        
        # Aggiungi Jolly alla mano
        self.player_hands[self.bot_player_id].append(joker)
        
        # Rimuovi carta usata per sostituzione
        self.player_hands[self.bot_player_id].remove(card)
        
        return self.POINTS_REPLACE_JOKER
    
    def _action_skip_meld(self) -> float:
        """Salta fase calata."""
        if self.game_phase != GamePhase.MELD:
            return -1.0
        self.game_phase = GamePhase.DISCARD
        return 0.0
    
    def _action_discard(self, card_idx: int) -> float:
        """Scarta una carta."""
        card = self.idx_to_card.get(card_idx)
        if not card or card not in self.player_hands[self.bot_player_id]:
            return -1.0
        
        self.player_hands[self.bot_player_id].remove(card)
        self.discard_pile.append(card)
        self.cards_seen[self.card_to_idx[card]] += 1
        
        # Passa al prossimo giocatore
        self.current_player = (self.current_player + 1) % self.num_players
        self.game_phase = GamePhase.DRAW
        self.bot_has_drawn = False
        self.bot_melded_this_turn = False
        self.turn_count += 1
        
        return 0.1
    
    def _action_close_round(self) -> float:
        """Chiude il round."""
        if not self.bot_melded_this_turn or self.player_hands[self.bot_player_id]:
            return -1.0
        
        self.round_over = True
        return self.REWARD_WIN
    
    def _simulate_opponent_turns(self):
        """Simula turni avversari con politica semplice (greedy)."""
        while self.current_player != self.bot_player_id and not self.round_over:
            player = self.current_player
            hand = self.player_hands[player]
            
            # Fase 1: Pesca (sempre dal tallone per semplicità)
            if self.stock_pile:
                card = self.stock_pile.pop()
                hand.append(card)
            elif self.discard_pile:
                # Se tallone vuoto, prendi dal pozzo
                card = self.discard_pile.pop()
                hand.append(card)
            else:
                # Nessuna carta disponibile
                break
            
            # Fase 2: Prova a calare (greedy)
            valid_melds = self._find_valid_melds(hand)
            for meld_cards in valid_melds:
                # Rimuovi carte
                for c in meld_cards:
                    if c in hand:
                        hand.remove(c)
                
                # Crea meld
                is_set = len(set(c.rank for c in meld_cards if c.rank != CardRank.JOKER)) == 1
                new_meld = Meld(
                    meld_id=len(self.table_melds),
                    meld_type='set' if is_set else 'run',
                    cards=meld_cards[:],
                    owner=player
                )
                self.table_melds.append(new_meld)
                
                # Prova ad attaccare altre carte
                for meld in self.table_melds:
                    for card in hand[:]:
                        if meld.can_attach(card):
                            hand.remove(card)
                            meld.cards.append(card)
            
            # Fase 3: Scarta (carta casuale)
            if hand:
                discard_card = hand[self.np_random.integers(len(hand))]
                hand.remove(discard_card)
                self.discard_pile.append(discard_card)
            
            # Passa al prossimo
            self.current_player = (self.current_player + 1) % self.num_players
            
            # Verifica se ha chiuso
            if not hand and any(m.owner == player for m in self.table_melds):
                self.round_over = True
                # Penalità per il bot
                break
    
    def _calculate_final_reward(self) -> float:
        """Calcola reward finale basato su chi ha chiuso."""
        if not self.round_over:
            return 0.0
        
        # Controlla se il bot ha chiuso (mano vuota + ha calato)
        bot_hand = self.player_hands[self.bot_player_id]
        bot_melded = any(m.owner == self.bot_player_id for m in self.table_melds)
        
        if len(bot_hand) == 0 and bot_melded:
            # Bot ha chiuso - vittoria!
            return self.REWARD_WIN
        
        # Un avversario ha chiuso - sconfitta
        # Calcola penalità basata su carte rimaste
        penalty = len(bot_hand) * self.PENALTY_CARD_IN_HAND
        return self.REWARD_LOSE - penalty
    
    def render(self):
        """Rendering testuale dello stato."""
        print("=" * 50)
        print(f"Turno: {self.turn_count}, Giocatore: {self.current_player}")
        print(f"Fase: {self.game_phase.name}")
        print(f"Tallone: {len(self.stock_pile)} carte")
        print(f"Pozzo: {[str(c) for c in self.discard_pile[-5:]]}")
        print(f"Giochi sul tavolo: {len(self.table_melds)}")
        for meld in self.table_melds:
            print(f"  - {meld}")
        print(f"Mano Bot ({len(self.player_hands[0])} carte): {[str(c) for c in self.player_hands[0]]}")
        for i in range(1, self.num_players):
            print(f"Mano P{i} ({len(self.player_hands[i])} carte)")
        print("=" * 50)


# Factory function per registration
def make_env():
    return PinnacolaEnv()
