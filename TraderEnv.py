import process_data
import pandas as pd
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from pathlib import Path
from timethis import timethis

# position constant
LONG = 0
SHORT = 1
FLAT = 2

# action constant
BUY = 0
SELL = 1
HOLD = 2


class OhlcvEnv(gym.Env):

    def __init__(self, window_size, path, show_trade=True):
        self.show_trade = show_trade
        self.path = path
        self.actions = ["LONG", "SHORT", "FLAT"]
        self.fee = 0.001
        self.seed()
        self.file_list = []
        # load_csv
        self.load_from_csv()

        # n_features
        self.window_size = window_size
        self.n_features = self.df.shape[1]
        self.shape = (self.window_size, self.n_features + 4)

        # defines action space
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

    def load_from_csv(self):
        if len(self.file_list) == 0:
            self.file_list = [x.name for x in Path(self.path).iterdir() if x.is_file()]
            self.file_list.sort()
        self.rand_episode = self.file_list.pop()
        raw_df = pd.read_csv(self.path + self.rand_episode)
        col = raw_df.shape[1]
        extractor = process_data.FeatureExtractor(raw_df)
        self.df = extractor.add_bar_features()  # bar features o, h, l, c ---> C(4,2) = 4*3/2*1 = 6 features
        self.df = extractor.add_mv_avg_features()
        self.df = extractor.add_adj_features()
        self.df = extractor.add_ta_features()

        self.df.dropna(inplace=True)  # drops Nan rows
        self.df['closingPrices'] = self.df['close']
        self.df = self.df.iloc[:, col:]
        self.closingPrices = self.df['close'].values
        self.df = self.df.values

    def render(self, mode='human', verbose=False):
        return None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # @timethis
    def step(self, action):
        if self.done:
            return self.state, self.reward, self.done, {}
        self.reward = 0

        # action comes from the agent
        # 0 buy, 1 sell, 2 hold
        # single position can be opened per trade
        # valid action sequence would be
        # LONG : buy - hold - hold - sell
        # SHORT : sell - hold - hold - buy
        # invalid action sequence is just considered hold
        # (e.g.) "buy - buy" would be considred "buy - hold"
        self.action = HOLD  # hold
        if action == BUY:  # buy
            if self.position == FLAT:  # if previous position was flat
                self.position = LONG  # update position to long
                self.action = BUY  # record action as buy
                self.entry_price = self.closingPrice  # maintain entry price
                self.n_long += 1  # record number of long
            elif self.position == SHORT:  # if previous position was short
                self.position = FLAT  # update position to flat
                self.action = BUY  # record action as buy
                # calculate reward
                self.reward = self.entry_price / self.closingPrice * (1 - self.fee) ** 2 - 1
                self.usdt_balance *= 1 + self.reward
                self.entry_price = 0  # clear entry price
        elif action == SELL:  # vice versa for short trade
            if self.position == FLAT:
                self.position = SHORT
                self.action = SELL
                self.entry_price = self.closingPrice
                self.n_short += 1
            elif self.position == LONG:
                self.position = FLAT
                self.action = SELL
                self.reward = self.closingPrice / self.entry_price * (1 - self.fee) ** 2 - 1
                self.usdt_balance *= 1 + self.reward
                self.entry_price = 0

        # [coin + usdt] total value evaluated in usdt
        if self.position == LONG:
            new_portfolio = self.usdt_balance * self.closingPrice / self.entry_price * (1 - self.fee) ** 2
        elif self.position == SHORT:
            new_portfolio = self.usdt_balance * self.entry_price / self.closingPrice * (1 - self.fee) ** 2
        else:
            new_portfolio = self.usdt_balance

        # 投资组合
        self.portfolio = new_portfolio
        self.current_tick += 1
        if self.show_trade and self.current_tick % 100 == 0:
            print("Tick: {0}/ Portfolio (usdt): {1}".format(self.current_tick, self.portfolio))
            print("Long: {0}/ Short: {1}".format(self.n_long, self.n_short))
        self.history.append((self.action, self.current_tick, self.closingPrice, self.portfolio, self.reward))
        self.updateState()
        if self.current_tick > self.df.shape[0] - self.window_size - 1:
            self.done = True
            self.reward = self.get_profit()  # return reward at end of the game
        return self.state, self.reward, self.done, {'portfolio': np.array([self.portfolio]),
                                                    "history": self.history,
                                                    "n_trades": {'long': self.n_long, 'short': self.n_short}}

    def get_profit(self):
        if self.position == LONG:
            return self.closingPrice / self.entry_price * (1 - self.fee) ** 2 - 1
        elif self.position == SHORT:
            return self.entry_price / self.closingPrice * (1 - self.fee) ** 2 - 1
        else:
            return 0

    def reset(self):
        # self.current_tick = random.randint(0, self.df.shape[0]-1000)
        self.current_tick = 0
        print("start episode ... {0} at {1}".format(self.rand_episode, self.current_tick))

        # positions
        self.n_long = 0
        self.n_short = 0

        # clear internal variables
        self.history = []  # keep buy, sell, hold action history
        self.usdt_balance = 100 * 10000  # initial balance, u can change it to whatever u like
        self.portfolio = self.usdt_balance  # (coin * current_price + current_usdt_balance) == portfolio
        self.profit = 0

        self.action = HOLD
        self.position = FLAT
        self.done = False

        # returns observed_features +  opened position(LONG/SHORT/FLAT) + profit_earned(during opened position)
        self.updateState()
        return self.state

    def updateState(self):
        def one_hot_encode(x, n_classes):
            return np.eye(n_classes)[x]

        self.closingPrice = self.closingPrices[self.current_tick]
        prev_position = self.position
        one_hot_position = one_hot_encode(prev_position, 3)
        profit = self.get_profit()
        # append two
        self.state = np.concatenate((self.df[self.current_tick], one_hot_position, [profit]))
        return self.state
