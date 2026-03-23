"""
Pinnacola RL Assistant - Environment Module
FASE 1: Environment Gymnasium per Pinnacola
"""

import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from enum import IntEnum
from dataclasses import dataclass, field
import itertools


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
        if card.rank == CardRank.JOKER:
            return False  # Non si attacca un Jolly direttamente
        if self.meld_type == 'set':
            # Tris: stesso rank, seme diverso da quelli già presenti
            non_joker = [c for c in self.cards if c.rank != CardRank.JOKER]
            if not non_joker:
                return False
            if card.rank != non_joker[0].rank:
                return False
            existing_suits = {c.suit for c in non_joker}
            return card.suit not in existing_suits and len(self.cards) < 4
        elif self.meld_type == 'run':
            # Scala: stesso seme, consecutiva (supporta Asso alto: Q-K-A)
            ranks = sorted([c.rank for c in self.cards if c.rank != CardRank.JOKER])
            if not ranks:
                return False
            suit = self.cards[0].suit if self.cards[0].rank != CardRank.JOKER else self.cards[1].suit
            if card.suit != suit:
                return False
            # Può estendere in basso
            if card.rank == ranks[0] - 1:
                return True
            # Può estendere in alto
            if card.rank == ranks[-1] + 1:
                return True
            # Asso alto: se la scala finisce con K (13), l'Asso (1) può chiudere
            if card.rank == CardRank.ACE and ranks[-1] == CardRank.KING:
                return True
            return False
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
    
    def __init__(self, num_players: int = 4, auto_simulate_opponents: bool = True):
        super().__init__()
        
        self.num_players = num_players
        self.bot_player_id = 0  # Il bot è sempre il giocatore 0
        self.auto_simulate_opponents = auto_simulate_opponents
        self.opponent_policy_fn = None
        
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
            # 2 Jolly per mazzo — deck_id unici per evitare collisioni hash
            cards.append(Card(rank=CardRank.JOKER, suit=CardSuit.NONE, deck_id=deck_id * 2))
            cards.append(Card(rank=CardRank.JOKER, suit=CardSuit.NONE, deck_id=deck_id * 2 + 1))
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
    
    def _get_observation(self, player_id: Optional[int] = None) -> np.ndarray:
        """Compone l'observation vector completo per un dato giocatore."""
        if player_id is None:
            player_id = self.current_player
            
        obs_parts = []
        
        # 1. Mano del giocatore (one-hot)
        bot_hand = self.player_hands[player_id]
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
        idx = 0
        for i in range(self.num_players):
            if i != player_id:
                opponent_cards[idx] = len(self.player_hands[i]) / self.MAX_HAND_SIZE
                idx += 1
        obs_parts.append(opponent_cards)
        
        # 6. Carte viste (memoria per probabilità)
        obs_parts.append(self.cards_seen[player_id].astype(np.float32) / self.NUM_DECKS)
        
        # 7. Fase del turno (one-hot)
        phase = np.zeros(4, dtype=np.float32)
        phase[self.game_phase] = 1
        obs_parts.append(phase)
        
        return np.concatenate(obs_parts)
    
    def _get_legal_actions(self, player_id: Optional[int] = None) -> List[Tuple[int, int, int, int]]:
        """
        Restituisce lista di azioni legali come tuple.
        Usato per generare la action mask.
        """
        if player_id is None:
            player_id = self.current_player
            
        legal = []
        bot_hand = self.player_hands[player_id]
        
        if self.game_phase == GamePhase.DRAW:
            # Le azioni DRAW non dipendono dalle carte in mano
            # (il bot deve poter pescare anche con mano vuota)
            if self.stock_pile:
                legal.append((ActionType.DRAW_STOCK, 0, 0, 0))
            if self.discard_pile:
                legal.append((ActionType.DRAW_PILE, 0, 0, 0))
                    
        elif self.game_phase == GamePhase.MELD:
            # Può calare combinazioni — un'azione per meld, distinguendo SET/RUN
            melds_possible = self._find_valid_melds(bot_hand)
            for meld_idx_local, meld_cards in enumerate(melds_possible):
                is_set = len(set(c.rank for c in meld_cards if c.rank != CardRank.JOKER)) == 1
                action_type = ActionType.MELD_SET if is_set else ActionType.MELD_RUN
                # Usa la prima carta non-Jolly come identificativo
                repr_card = next((c for c in meld_cards if c.rank != CardRank.JOKER), meld_cards[0])
                legal.append((action_type, self.card_to_idx[repr_card], 0, len(meld_cards)))
            
            # Può attaccare carte ai giochi esistenti
            for meld_idx, meld in enumerate(self.table_melds):
                for card in bot_hand:
                    if meld.can_attach(card):
                        legal.append((ActionType.ATTACH_CARD, self.card_to_idx[card], meld_idx, 0))
            
            # Può sostituire Jolly su QUALSIASI meld (non solo propri)
            for meld_idx, meld in enumerate(self.table_melds):
                if meld.has_joker():
                    for card in bot_hand:
                        if self._can_replace_joker(meld, card):
                            legal.append((ActionType.REPLACE_JOKER, self.card_to_idx[card], meld_idx, 0))
            
            # Può saltare la fase
            legal.append((ActionType.SKIP_MELD, 0, 0, 0))
            
        elif self.game_phase == GamePhase.DISCARD:
            # Deve scartare una carta dalla mano
            for card in bot_hand:
                legal.append((ActionType.DISCARD, self.card_to_idx[card], 0, 0))
            # Può chiudere se mano vuota e ha calato almeno un gioco in totale
            has_melded_ever = any(m.owner == player_id for m in self.table_melds)
            if len(bot_hand) == 0 and has_melded_ever:
                legal.append((ActionType.CLOSE_ROUND, 0, 0, 0))
        
        return legal
    
    def _find_valid_melds(self, hand: List[Card]) -> List[List[Card]]:
        """Trova tutte le combinazioni valide (tris/scale) nella mano."""
        valid_melds = []
        
        import itertools
        
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
            # Raggruppa per seme per evitare duplicati dello stesso seme nel tris
            suits_dict = {}
            for c in cards:
                if c.suit not in suits_dict:
                    suits_dict[c.suit] = []
                suits_dict[c.suit].append(c)
            
            available_suits = list(suits_dict.keys())
            
            # Tris senza Jolly: almeno 3 semi diversi (3 o 4 carte)
            for k in range(3, min(len(available_suits) + 1, 5)):
                for subset_suits in itertools.combinations(available_suits, k):
                    for combo in itertools.product(*(suits_dict[s] for s in subset_suits)):
                        valid_melds.append(list(combo))
            
            # Tris con Jolly: 2 semi DIVERSI + 1 Jolly = tris (size 3)
            # NB: non serve quartetto con jolly (4 semi coprono già tutto)
            if num_jokers >= 1 and len(available_suits) >= 2:
                for subset_suits in itertools.combinations(available_suits, 2):
                    for combo in itertools.product(*(suits_dict[s] for s in subset_suits)):
                        valid_melds.append(list(combo) + [jokers[0]])
        
        # Trova scale (3+ carte consecutive, stesso seme)
        for suit in range(4):  # Per ogni seme
            rank_map = {}
            for c in hand:
                if c.suit == CardSuit(suit) and c.rank != CardRank.JOKER:
                    if c.rank not in rank_map:
                        rank_map[c.rank] = []
                    rank_map[c.rank].append(c)
            
            if not rank_map:
                continue
                
            ranks_present = sorted(list(rank_map.keys()))
            
            # Trova sequenze senza Jolly
            for i in range(len(ranks_present)):
                for j in range(i + 3, len(ranks_present) + 1):
                    subset_ranks = ranks_present[i:j]
                    # Verifica se sono perfettamente consecutivi
                    if subset_ranks[-1] - subset_ranks[0] == len(subset_ranks) - 1:
                        for combo in itertools.product(*(rank_map[r] for r in subset_ranks)):
                            valid_melds.append(list(combo))
            
            # Scala Asso-alto senza Jolly: ..., Q, K, A
            if CardRank.ACE in rank_map:
                # Cerca sequenze che finiscono con K (13) e aggiungi A
                for i in range(len(ranks_present)):
                    for j in range(i + 2, len(ranks_present) + 1):
                        subset_ranks = ranks_present[i:j]
                        if CardRank.ACE in subset_ranks:
                            continue  # A è già nella sequenza come basso
                        if subset_ranks[-1] == CardRank.KING and subset_ranks[-1] - subset_ranks[0] == len(subset_ranks) - 1:
                            # Sequenza valida che finisce con K, aggiungi A
                            for combo_base in itertools.product(*(rank_map[r] for r in subset_ranks)):
                                for ace_card in rank_map[CardRank.ACE]:
                                    valid_melds.append(list(combo_base) + [ace_card])
            
            # Trova sequenze con 1 Jolly
            if num_jokers >= 1:
                for i in range(len(ranks_present)):
                    for j in range(i + 2, len(ranks_present) + 1):
                        subset_ranks = ranks_present[i:j]
                        gap = subset_ranks[-1] - subset_ranks[0]
                        # "Buco" di una carta tra gli estremi -> Scala valida con inserimento del Jolly
                        if gap == len(subset_ranks):
                            for combo in itertools.product(*(rank_map[r] for r in subset_ranks)):
                                valid_melds.append(list(combo) + [jokers[0]])
                        # Estremi consecutivi (es 3-4 e Jolly forma J-3-4 o 3-4-J)
                        elif gap == len(subset_ranks) - 1:
                            for combo in itertools.product(*(rank_map[r] for r in subset_ranks)):
                                valid_melds.append(list(combo) + [jokers[0]])
            
            # Trova sequenze con 2 Jolly
            if num_jokers >= 2:
                for i in range(len(ranks_present)):
                    for j in range(i + 1, len(ranks_present) + 1):
                        subset_ranks = ranks_present[i:j]
                        num_real = len(subset_ranks)
                        span = subset_ranks[-1] - subset_ranks[0] + 1
                        gaps = span - num_real
                        total_len = span  # lunghezza della scala risultante
                        # Serve esattamente 2 jolly per i buchi, e scala almeno 3
                        if gaps == 2 and total_len >= 3:
                            for combo in itertools.product(*(rank_map[r] for r in subset_ranks)):
                                valid_melds.append(list(combo) + jokers[:2])
                        # 1 buco interno + 1 jolly per estendere (sopra o sotto)
                        elif gaps == 1 and total_len + 1 >= 3:
                            for combo in itertools.product(*(rank_map[r] for r in subset_ranks)):
                                valid_melds.append(list(combo) + jokers[:2])
                        # 0 buchi interni + 2 jolly per estendere
                        elif gaps == 0 and num_real >= 1 and num_real + 2 >= 3:
                            for combo in itertools.product(*(rank_map[r] for r in subset_ranks)):
                                valid_melds.append(list(combo) + jokers[:2])
        
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
            non_joker = [c for c in meld.cards if c.rank != CardRank.JOKER]
            if not non_joker:
                return False
            if card.rank != non_joker[0].rank or card.rank == CardRank.JOKER:
                return False
            # Verifica che il seme non sia già presente
            existing_suits = {c.suit for c in non_joker}
            return card.suit not in existing_suits
        elif meld.meld_type == 'run':
            # In una scala, il Jolly rappresenta un rank specifico nella sequenza
            ranks = sorted([c.rank for c in meld.cards if c.rank != CardRank.JOKER])
            suits = [c.suit for c in meld.cards if c.suit != CardSuit.NONE]
            if not ranks or not suits:
                return False
            
            suit = suits[0]
            if card.suit != suit:
                return False
            
            # Il Jolly può essere in qualsiasi posizione gap della scala
            possible_gaps = []
            for i in range(len(ranks) - 1):
                if ranks[i+1] - ranks[i] > 1:
                    possible_gaps.extend(range(ranks[i] + 1, ranks[i+1]))
            
            # Verifica anche Asso alto (rank 1 sostituisce dopo K=13)
            possible_ranks = possible_gaps
            if ranks[-1] + 1 == CardRank.JOKER:
                # ranks[-1] è 13 (K), il jolly potrebbe rappresentare l'Asso alto
                if card.rank == CardRank.ACE:
                    return True
            return card.rank in possible_ranks
        
        return False
    
    def _get_action_mask(self, player_id: Optional[int] = None) -> np.ndarray:
        """Genera maschera booleana per azioni legali."""
        mask = np.zeros(self._get_action_space_size(), dtype=np.int8)
        legal_actions = self._get_legal_actions(player_id)
        
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
        self.cards_seen = np.zeros((self.num_players, self.TOTAL_CARDS), dtype=np.int32)
        
        # Registra carte già viste per ogni giocatore
        for p in range(self.num_players):
            for card in self.player_hands[p]:
                self.cards_seen[p, self.card_to_idx[card]] += 1
            for card in self.discard_pile:
                self.cards_seen[p, self.card_to_idx[card]] += 1
        
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
        if self.auto_simulate_opponents and self.current_player != self.bot_player_id and not self.round_over:
            self._simulate_opponent_turns()
        
        # Stalemate: se tallone e pozzo vuoti, la partita finisce
        if not self.stock_pile and not self.discard_pile and not self.round_over:
            self.round_over = True
        
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
        self.player_hands[self.current_player].append(card)
        self.cards_seen[self.current_player, self.card_to_idx[card]] += 1
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
        self.player_hands[self.current_player].extend(cards_to_take)
        
        self.bot_has_drawn = True
        self.game_phase = GamePhase.MELD
        return 0.1
    
    def _action_meld(self, meld_type: int, card_idx: int, num_cards: int) -> float:
        """Cala una combinazione."""
        if self.game_phase != GamePhase.MELD:
            return -1.0
        
        # Trova la combinazione valida che corrisponde
        bot_hand = self.player_hands[self.current_player]
        valid_melds = self._find_valid_melds(bot_hand)
        
        if not valid_melds:
            return -1.0  # Nessuna combinazione valida
        
        target_card = self.idx_to_card.get(card_idx)
        
        # Cerca la combinazione che corrisponde a target_card e num_cards
        meld_cards = None
        for m in valid_melds:
            if len(m) == num_cards and target_card in m:
                meld_cards = m
                break
                
        if not meld_cards:
            # Fallback (non dovrebbe verificarsi in un mapping corretto azione-stato)
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
            owner=self.current_player
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
        if not card or card not in self.player_hands[self.current_player]:
            return -1.0
        
        meld = self.table_melds[meld_idx]
        
        if not meld.can_attach(card):
            return -1.0
        
        # Rimuovi carta dalla mano e aggiungi al meld
        self.player_hands[self.current_player].remove(card)
        meld.cards.append(card)
        
        return self.POINTS_ATTACH
    
    def _action_replace_joker(self, card_idx: int, meld_idx: int) -> float:
        """Sostituisce un Jolly."""
        if self.game_phase != GamePhase.MELD:
            return -1.0
        
        if meld_idx >= len(self.table_melds):
            return -1.0
        
        card = self.idx_to_card.get(card_idx)
        if not card or card not in self.player_hands[self.current_player]:
            return -1.0
        
        meld = self.table_melds[meld_idx]
        
        if not self._can_replace_joker(meld, card):
            return -1.0
        
        # Trova e rimuovi il Jolly
        joker = meld.get_replaceable_joker()
        if not joker:
            return -1.0
        
        meld.cards.remove(joker)
        meld.cards.append(card)
        
        # Aggiungi Jolly alla mano
        self.player_hands[self.current_player].append(joker)
        
        # Rimuovi carta usata per sostituzione
        self.player_hands[self.current_player].remove(card)
        
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
        if not card or card not in self.player_hands[self.current_player]:
            return -1.0
        
        self.player_hands[self.current_player].remove(card)
        self.discard_pile.append(card)
        
        for p in range(self.num_players):
            if p != self.current_player:
                self.cards_seen[p, self.card_to_idx[card]] += 1
        
        # Passa al prossimo giocatore
        self.current_player = (self.current_player + 1) % self.num_players
        self.game_phase = GamePhase.DRAW
        self.bot_has_drawn = False
        self.bot_melded_this_turn = False
        self.turn_count += 1
        
        return 0.1
    
    def _action_close_round(self) -> float:
        """Chiude il round."""
        has_melded = any(m.owner == self.current_player for m in self.table_melds)
        if not has_melded or self.player_hands[self.current_player]:
            return -1.0
        
        self.round_over = True
        return self.REWARD_WIN
    
    def _simulate_opponent_turns(self):
        """
        Simula turni avversari con politica semi-random realistica.
        Ogni avversario: pesca 1, cala al massimo 1 combinazione,
        attacca con probabilità 50%, scarta 1.
        """
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
                # Nessuna carta disponibile — stalemate
                self.round_over = True
                break
            
            # Fase 2: Prova a calare AL MASSIMO 1 combinazione (realistico)
            valid_melds = self._find_valid_melds(hand)
            if valid_melds:
                # Sceglie una combinazione a caso
                meld_cards = valid_melds[self.np_random.integers(len(valid_melds))]
                
                # Verifica che tutte le carte siano ancora in mano
                can_play = all(c in hand for c in meld_cards)
                if can_play:
                    for c in meld_cards:
                        hand.remove(c)
                    
                    is_set = len(set(c.rank for c in meld_cards if c.rank != CardRank.JOKER)) == 1
                    new_meld = Meld(
                        meld_id=len(self.table_melds),
                        meld_type='set' if is_set else 'run',
                        cards=meld_cards[:],
                        owner=player
                    )
                    self.table_melds.append(new_meld)
            
            # Fase 2b: Prova ad attaccare UNA carta (50% probabilità)
            if hand and self.np_random.random() > 0.5:
                for meld in self.table_melds:
                    attached = False
                    for card in hand[:]:
                        if meld.can_attach(card):
                            hand.remove(card)
                            meld.cards.append(card)
                            attached = True
                            break  # Al massimo 1 attach per turno
                    if attached:
                        break
            
            # Fase 3: Scarta (carta casuale)
            if hand:
                discard_card = hand[self.np_random.integers(len(hand))]
                hand.remove(discard_card)
                self.discard_pile.append(discard_card)
                # Aggiorna cards_seen per tutti gli altri giocatori (vedono lo scarto)
                for p in range(self.num_players):
                    if p != player:
                        self.cards_seen[p, self.card_to_idx[discard_card]] += 1
            
            # Incrementa il contatore turni per ogni giocatore
            self.turn_count += 1
            
            # Passa al prossimo
            self.current_player = (self.current_player + 1) % self.num_players
            
            # Verifica se ha chiuso (mano vuota + almeno un gioco calato)
            if not hand and any(m.owner == player for m in self.table_melds):
                self.round_over = True
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
