import numpy as np


class BlackJackEnv():
    def __init__(self, num_decks=1, natural_blackjack=True):
        """
        Initialize the Blackjack environment.
        
        :param num_decks: Number of decks to use in the game.
        :param natural_blackjack: Whether to allow natural blackjack (21 with first two cards).
        """
        self.num_decks = num_decks
        self.natural_blackjack = natural_blackjack
        self.action_space = 2
        self.observation_space = (32, 11, 2)
        self.deck = self.create_deck()
        self.player_hand = []
        self.dealer_hand = []
        self.player_sum = 0
        self.dealer_sum = 0
        self.usable_ace = False
        self.reset()

    def create_deck(self):
        """
        Create a deck of cards for the Blackjack game.
        
        :return: A list representing the deck of cards.
        """
        # 1-10 are numbered cards, 11-13 are face cards (Jack, Queen, King)
        deck = [i for i in range(1, 14)] * 4 * self.num_decks
        return deck
    
    def shuffle_deck(self):
        """
        Shuffle the deck of cards.
        """
        np.random.shuffle(self.deck)
    
    def reset(self):
        """
        Reset the environment to the initial state.
        
        :return: The initial state of the environment.
        """
        self.player_hand = []
        self.dealer_hand = []
        self.player_sum = 0
        self.dealer_sum = 0
        self.usable_ace = False
        self.deck = self.create_deck()
        self.deal_initial_cards()
        return self.get_state(), 1 # The second value is a dummy for compatibility with the agent
    
    def deal_initial_cards(self):
        """
        Deal the initial two cards to both the player and the dealer.
        """
        self.shuffle_deck()
        self.player_hand = [self.draw_card(), self.draw_card()]
        self.dealer_hand = [self.draw_card(), self.draw_card()]
        self.update_sums()

    def draw_card(self):
        """
        Draw a card from the deck.
        
        :return: The drawn card.
        """
        if not self.deck:
            self.deck = self.create_deck()
            self.shuffle_deck()
        return self.deck.pop()
    
    def update_sums(self):
        """
        Update the sums of the player's and dealer's hands.
        Also check if the player has a usable ace.
        """
        self.player_sum = 0
        self.usable_ace = False
        for card in self.player_hand:
            if card == 1:
                self.player_sum += 1
            if card > 10:
                self.player_sum += 10
            else:
                self.player_sum += card

        for card in self.player_hand:
            if card == 1 and self.player_sum + 10 <= 21:
                self.player_sum += 10
                self.usable_ace = True

        self.dealer_sum = 0
        for card in self.dealer_hand:
            if card == 1:
                self.dealer_sum += 1
            if card > 10:
                self.dealer_sum += 10
            else:
                self.dealer_sum += card

        for card in self.dealer_hand:
            if card == 1 and self.dealer_sum + 10 <= 21:
                self.dealer_sum += 10
        


    def get_state(self):
        """
        Get the current state of the environment.
        
        :return: A tuple representing the state (player_sum, dealer_card, usable_ace).
        """
        dealer_card = self.dealer_hand[0] if self.dealer_hand[0] != 1 else 11  # Treat Ace as 11 for dealer's visible card
        if dealer_card > 10:
            dealer_card = 10
        return (self.player_sum, dealer_card, int(self.usable_ace)) # The second value is a dummy for compatibility with the agent
    

    def step(self, action):
        """
        Take a step in the environment based on the action.
        
        :param action: The action to take (0 for 'hit', 1 for 'stand').
        :return: A tuple containing the next state, reward, done flag, and additional info.
        """
        if action == 0:
            # Player chooses to stick
            self.dealer_deals()
            reward = self.calculate_reward()
            done = True
        elif action == 1:
            # Player chooses to hit
            self.player_hand.append(self.draw_card())
            self.update_sums()
            if self.player_sum > 21:
                reward = -1
                done = True
            else:
                reward = 0
                done = False
        
        else:
            raise ValueError("Invalid action. Action must be 0 (hit) or 1 (stand).")
        
        return self.get_state(), reward, done, {}, {}
    
    def dealer_deals(self):
        """
        Dealer plays according to the rules of Blackjack.
        Dealer must hit until their sum is at least 17.
        """
        while self.dealer_sum < 17:
            self.dealer_hand.append(self.draw_card())
            self.update_sums()

    def calculate_reward(self):
        """
        Calculate the reward based on the final sums of the player and dealer.
        
        :return: The reward for the player.
        """
        if self.player_sum > 21:
            return -1
        elif self.dealer_sum > 21 or self.player_sum > self.dealer_sum:
            if self.natural_blackjack and self.player_sum == 21 and len(self.player_hand) == 2:
                return 1.5
            return 1
        elif self.player_sum < self.dealer_sum:
            return -1
        else:
            return 0