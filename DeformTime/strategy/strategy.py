from abc import ABC, abstractmethod
import numpy as np
import sys
import pandas as pd
from portfolio import Portfolio
import random
import torch

class Strategy(ABC):
    @staticmethod
    def initialize(args, **kwargs):
        if args.strategy == 'bandit':
            return BanditStrategy(args, **kwargs)
        elif args.strategy == 'b2':
            return Bandit2Strategy(args, **kwargs)
        elif args.strategy == 'neural':
            return NeuralNetworkStrategy(args, **kwargs)

    @abstractmethod
    def make_move(self, context=None, forecast=None, **kwargs):
        pass

    @abstractmethod
    def report_reward(self, reward):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def reset(self):
        pass

# class BanditStrategy(Strategy):
#     def __init__(self, args, **kwargs):
#         self.args = args
#         #self.advisors = args.advisors
#         self.long_bandit = Bandit.initialize(args)
#         self.short_bandit = Bandit.initialize(args)

#         self.scaler, self.metric = None, None

#         if args.scaler == 'MA':
#             self.scaler = moving_average_error_scaler

#         self.ranker = Ranker()

#         self.stocks = {'long': {}, 'short': {}}
#         self.purchased_shares = {'long': {}, 'short': {}}
#         self.bought_price = {'long': {}, 'short': {}}

#         self.companies = kwargs['stocks']

#         self.history = {}
#         self.running_errors = {}
#         for company in self.companies:
#             self.history[company] = []
#             self.running_errors[company] = 0.0

#     def make_move(self, context=None, forecast=None, **kwargs):
#         #money_to_invest = kwargs['invest']

#         # Advisors
#         # self.args.advisors = [
#         #     np.random.standard_gamma(1, size=30),
#         #     np.random.uniform(0, 1, self.args.bandit_arms)
#         # ]

#         # self.args.advisors = np.array(self.args.advisors)
#         self.args.advisors = None

#         self.money_pool = self.args.initial_money

#         returns = context.filter(regex=".*_predicted_RET")

#         # GO Long
#         # Buy then calculate return on next day
#         # Sort by expected return
#         #long_returns = returns.sort_values(ascending=False)
#         long_returns = self.ranker.scaled_rank(returns.copy(), confidence=self.metric, scaler=self.scaler, sort='descending', window_size=self.args.window_size)
#         if self.metric is not None:
#             self.metric = 1 / self.metric
#         advisors = np.array([1/self.args.bandit_arms for _ in range(self.args.bandit_arms)]) if self.metric is None else softmax((self.metric[:self.args.bandit_arms]))
#         if len(advisors.shape) == 1:
#             advisors = np.expand_dims(advisors, axis=0)

#         self.how_many_to_long = self.long_bandit.select_arm(context=context, advisors = advisors) ########

#         # GO Short
#         # Sell and then calculate return
#         #short_returns = returns.sort_values(ascending=True)
#         short_returns = self.ranker.scaled_rank(returns.copy(), confidence=self.metric, scaler=self.scaler, sort='ascending', window_size=self.args.window_size)
#         advisors = np.array([1/self.args.bandit_arms for _ in range(self.args.bandit_arms)]) if self.metric is None else softmax((self.metric[:self.args.bandit_arms]))
#         if len(advisors.shape) == 1:
#             advisors = np.expand_dims(advisors, axis=0)

#         self.how_many_to_short = self.short_bandit.select_arm(context=context, advisors = advisors) ########

#         try:
#             self.budget_for_long =  (self.how_many_to_long * self.money_pool) //  (self.how_many_to_long + self.how_many_to_short)
#             self.budget_for_short = (self.how_many_to_short * self.money_pool) //  (self.how_many_to_long + self.how_many_to_short)
#         except:
#             self.budget_for_long = 0
#             self.budget_for_short = 0
            
#         if self.budget_for_short > 0:
#             selected_stocks_to_long = long_returns.iloc[:self.how_many_to_long]
#             money_per_stock = self.budget_for_long // self.how_many_to_long

#             for index, _ in selected_stocks_to_long.items():
#                 idx = index.split('_')[0]
#                 stock_price = context[idx + '_PRC']
#                 shares = money_per_stock / stock_price
#                 self.stocks['long'][idx] = idx
#                 self.purchased_shares['long'][idx] = shares
#                 self.bought_price['long'][idx] = stock_price

#                 #if len(self.history[idx]) == self.args.window_size:
#                 #    self.history[idx].pop(0)
#                 #self.history[idx].append((forecast[f"{idx}_expected_RET"] - context[f"{idx}_predicted_RET"]))

#             self.money_pool -= self.budget_for_long

#         if self.budget_for_short > 0:

#             selected_stocks_to_short = short_returns.iloc[:self.how_many_to_short]
#             for index, _ in selected_stocks_to_short.items():
#                 idx = index.split('_')[0]
#                 if idx in self.stocks['long'].keys():
#                     self.how_many_to_short -= 1
#                     selected_stocks_to_short = selected_stocks_to_short.drop(labels=[f"{idx}_predicted_RET"])
            
#             if self.how_many_to_short != 0:
#                 money_per_stock = self.budget_for_short // self.how_many_to_short

#             for index, _ in selected_stocks_to_short.items():
#                 idx = index.split('_')[0]
#                 stock_sell_price = context[idx + '_SELL_PRC']
#                 shares = money_per_stock / stock_sell_price
#                 #gain = shares * stock_sell_price

#                 self.stocks['short'][idx] = idx
#                 self.purchased_shares['short'][idx] = shares
#                 self.bought_price['short'][idx] = stock_sell_price

#                 #if len(self.history[idx]) == self.args.window_size:
#                 #    self.history[idx].pop(0)
#                 #self.history[idx].append((forecast[f"{idx}_expected_RET"] - context[f"{idx}_predicted_RET"]))

#             self.money_pool -= self.budget_for_short

#         for company in self.companies:
#             self.history[company].append((forecast[f"{company}_expected_RET"] - context[f"{company}_predicted_RET"]))

#         return self.how_many_to_long, self.how_many_to_short if self.budget_for_short > 0 else 0

#     def report_reward(self, reward):
#         if self.metric is None:
#             self.metric = {}
#             for company in self.companies:
#                 self.metric[f"{company}_predicted_RET"] = 0.0
#             self.metric = pd.Series(self.metric)

#         for key, values in self.history.items():
#             self.running_errors[key] = self.args.ma_alpha * self.running_errors[key] + (1 - self.args.ma_alpha) * values[-1]
#             self.metric[f"{key}_predicted_RET"] = self.running_errors[key]
#             #mae(values) if len(values) > 0 else 0.0

#         self.long_bandit.update(reward=reward, choice=self.how_many_to_long)
#         self.short_bandit.update(reward=reward, choice=self.how_many_to_short)

#     def get_state(self):
#         portfolio = Portfolio(self.money_pool)
#         portfolio.add_stock(self.purchased_shares)
#         return portfolio
    
#     def reset(self):
#         self.stocks = {'long': {}, 'short': {}}
#         self.purchased_shares = {'long': {}, 'short': {}}
#         self.bought_price = {'long': {}, 'short': {}}


# class Bandit2Strategy(Strategy):
#     def __init__(self, args, **kwargs):
#         self.args = args
#         #self.advisors = args.advisors
#         self.bandit = Bandit.initialize(args)

#         self.scaler, self.metric = None, None

#         if args.scaler == 'MA':
#             self.scaler = moving_average_error_scaler

#         self.ranker = Ranker()

#         self.stocks = {'long': {}, 'short': {}}
#         self.purchased_shares = {'long': {}, 'short': {}}
#         self.bought_price = {'long': {}, 'short': {}}

#         self.companies = kwargs['stocks']

#         self.history = {}
#         self.running_errors = {}
#         for company in self.companies:
#             self.history[company] = []
#             self.running_errors[company] = 0.0

#         self.choices = {
#             0: "long",
#             1: "short",
#             "long": 0,
#             "short": 1
#         }

#     def make_move(self, context=None, forecast=None, **kwargs):
#         #money_to_invest = kwargs['invest']

#         # Advisors
#         # self.args.advisors = [
#         #     np.random.standard_gamma(1, size=30),
#         #     np.random.uniform(0, 1, self.args.bandit_arms)
#         # ]

#         # self.args.advisors = np.array(self.args.advisors)
#         #self.args.advisors = None

#         returns = context.filter(regex=".*_predicted_RET")

#         long_returns = self.ranker.scaled_rank(returns.copy(), confidence=self.metric, scaler=self.scaler, sort='descending', window_size=self.args.window_size)
#         short_returns = self.ranker.scaled_rank(returns.copy(), confidence=self.metric, scaler=self.scaler, sort='ascending', window_size=self.args.window_size)
        
#         # Advisors - 3
#         advisors = []
#         tmp = long_returns.copy()
#         for i in range(self.args.advisors):
#             advisor = []
#             for _ in range(self.args.bandit_arms):
#                 if i == 0:
#                     advisor.extend([1 / self.args.bandit_arms])
#                 elif i == 1:
#                     advisor.append(len(tmp[tmp > 0]))
#                     tmp = -1 * tmp
#             if i == 1:
#                 advisor = [a/np.sum(advisor) for a in advisor]
#             elif i == 2:
#                 advisor.extend(np.abs([tmp.values[0], tmp.values[-1]])/np.sum(np.abs([tmp.values[0], tmp.values[-1]])))
#             elif i == 3:
#                 if long_returns[0] > abs(long_returns[-1]):
#                     advisor.extend([1.0, 0.0])
#                 else:
#                     advisor.extend([0.0, 1.0])
#             advisors.append(advisor)
#         advisors = np.array(advisors)

#         #print(advisors)

#         self.choice = self.bandit.select_arm(context=context, advisors = advisors)

#         self.money_pool = self.args.initial_money

#         ranked_returns = long_returns if self.choices[self.choice] == "long" else short_returns

#         # if abs(long_returns[0]) > abs(short_returns[0]):
#         #     self.choice = self.choices["long"]
#         # else:
#         #     self.choice = self.choices["short"]

#         self.args.long_limit = random.randint(1, self.args.long_limit+1)
#         self.args.short_limit = random.randint(1, self.args.short_limit+1)

#         if self.choices[self.choice] == "long":
#             self.selected_stocks = ranked_returns.iloc[:self.args.long_limit]
#         elif self.choices[self.choice] == "short":
#             self.selected_stocks = ranked_returns.iloc[:self.args.short_limit]
        
#         money_per_stock = self.money_pool // len(self.selected_stocks)

#         for index, _ in self.selected_stocks.items():
#             idx = index.split('_')[0]
#             stock_price = context[idx + '_PRC'] if self.choices[self.choice] == 'long' else context[idx + '_SELL_PRC']
#             shares = money_per_stock / stock_price
#             self.stocks[self.choices[self.choice]][idx] = idx
#             self.purchased_shares[self.choices[self.choice]][idx] = shares
#             self.bought_price[self.choices[self.choice]][idx] = stock_price


#         self.money_pool = 0

#         for company in self.companies:
#             self.history[company].append((forecast[f"{company}_expected_RET"] - context[f"{company}_predicted_RET"]))

#         return len(self.selected_stocks), self.choices[self.choice]

#     def report_reward(self, reward):
#         if self.metric is None:
#             self.metric = {}
#             for company in self.companies:
#                 self.metric[f"{company}_predicted_RET"] = 0.0
#             self.metric = pd.Series(self.metric)

#         for key, values in self.history.items():
#             self.running_errors[key] = self.args.ma_alpha * self.running_errors[key] + (1 - self.args.ma_alpha) * values[-1]
#             self.metric[f"{key}_predicted_RET"] = self.running_errors[key]
#             #mae(values) if len(values) > 0 else 0.0

#         self.bandit.update(reward=reward, choice=self.choice)

#     def get_state(self):
#         portfolio = Portfolio(self.money_pool)
#         portfolio.add_stock(self.purchased_shares)
#         return portfolio
    
#     def reset(self):
#         self.stocks = {'long': {}, 'short': {}}
#         self.purchased_shares = {'long': {}, 'short': {}}
#         self.bought_price = {'long': {}, 'short': {}}



class NeuralNetworkStrategy(Strategy):
    def __init__(self, args, **kwargs):
        self.args = args
        self.metric = None, None

        self.stocks = {'long': {}, 'short': {}}
        self.purchased_shares = {'long': {}, 'short': {}}
        self.bought_price = {'long': {}, 'short': {}}

        #self.companies = kwargs['stocks']

        # self.history = {}
        # self.running_errors = {}
        # for company in self.companies:
        #     self.history[company] = []
        #     self.running_errors[company] = 0.0

        self.money_pool = self.args.money_pool

    def make_move(self, context=None, forecast=None, **kwargs):
        '''
        context - stock prices at time t
        forecast - stock prices at time t+1
        kwargs
            - Parameter "decision" which is the NN output layer V_i(s)
            - Parameter "stocks" will have the sequence of stocks like V_i(s)
        '''

        # print(context)
        # print(forecast)

        nn_output = kwargs["decision"]

        #returns = context.filter(regex=".*_RET")
        proportion = torch.abs(nn_output) / torch.sum(torch.abs(nn_output)).item()
        # print(f"proportion: {proportion}")

        assert torch.abs(torch.sum(proportion) - 1.0) < 0.0001
        
        nn_output = list(nn_output.detach().numpy().flatten())
        print(f"nn_output: {nn_output}")
        proportion = list(proportion.detach().numpy().flatten())
        print(f"v_i_cap: : {proportion}")

        companies = kwargs["stocks"]
        companies.append(None)
        print(f"companies for v_caps: {companies}")
        investment, pool = 0.0, self.money_pool
        decision, price_type = None, None

        # if not all(abs(proportion[-1]) > abs(proportion[:-1])):

        for st, v, p in zip(companies, nn_output, proportion):
            if st is None:
                if self.args.enable_action and all(abs(p) > abs(a) for a in proportion[:-1]):
                    self.reset()
                    decision = 'hold'
                continue
                    
            if v < 0: # short
                price_type = '_PRC'
                decision = 'short'
            elif v > 0: # long
                price_type = '_PRC'
                decision = 'long'
            # else:
            #     price_type = '_PRC'
            #     decision = 'hold'

            stock_price = context[st + price_type]
            # print(f"[INFO]              {st} stock price is {stock_price}")
            shares = p * self.money_pool / stock_price
            # print(f"[INFO]              {st} number of shares to trade is {shares}")
            investment += p * self.money_pool
            # self.money_pool -= p * self.money_pool
            self.stocks[decision][st] = st
            self.purchased_shares[decision][st] = shares
            self.bought_price[decision][st] = stock_price

            # print(f"[INFO]          For stock {st} go {decision} with {shares} shares | PRC_t: { context[st + price_type]} PRC_t+1: { forecast[st + price_type]}")


        # assert abs(investment - self.money_pool) < 1
            
        pool -= investment
        self.context = context
        print(f"[INFO]      Remaining Pool: {pool}")

    def report_reward(self, reward):
        self.money_pool += reward

    def get_state(self):
        print(f"[INFO]      Starting with {self.money_pool}")
        portfolio = Portfolio(self.money_pool, self.context)
        portfolio.add_stock(self.purchased_shares)
        return portfolio
    
    def reset(self):
        self.stocks = {'long': {}, 'short': {}}
        self.purchased_shares = {'long': {}, 'short': {}}
        self.bought_price = {'long': {}, 'short': {}}