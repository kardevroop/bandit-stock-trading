from abc import ABC, abstractmethod
import sys
sys.path.append('../')
from portfolio import Portfolio

class Oracle(ABC):
    @staticmethod
    def initialize(args):
        if args.oracle == 'portfolio':
            return PortFolioOracle(args.initial_money)
        elif args.oracle == 'neural':
            return NeuralPortFolioOracle(args.money_pool)
    @abstractmethod
    def calculate_reward(self, state, context):
        pass

    @abstractmethod
    def reset(self, args):
        pass

'''
Simple Portfolio Oracle
'''
class PortFolioOracle(Oracle):
    def __init__(self, initial_money_pool = 100.0):
        self.initial_money_pool = initial_money_pool
        self.money_pool = initial_money_pool
        self.previous_total_money = self.initial_money_pool

    def calculate_reward(self, state: Portfolio, context: dict):
        current_total_money = state.calculate_value(context)
        print(f"[INFO   ] Investment : {current_total_money}")
        reward = current_total_money - self.previous_total_money
        self.money_pool = current_total_money

        return reward, self.money_pool

    def reset(self, args):
        self.previous_total_money = self.money_pool

'''
Neural Portfolio Oracle
Calculates reward based on decision and stock price valuation
'''
class NeuralPortFolioOracle(Oracle):
    def __init__(self, initial_money_pool = 100.0):
        self.initial_money_pool = initial_money_pool
        self.money_pool = initial_money_pool
        self.previous_total_money = self.initial_money_pool

    def calculate_reward(self, state: Portfolio, context: dict):
        '''
        Portfolio containing value of stock bought/shorted
        context to calculate stock valuation on specific day
        '''
        portfolio_values, stock_shares, prev_context = state.calculate_value(context)
        reward = 0.0
        for decision in portfolio_values.keys():
            price = '_PRC' # if decision == 'short' else '_PRC'
            for stock in portfolio_values[decision].keys():
                # print(f"[INFO]   {decision}   {stock} price before: {portfolio_values[decision][stock]/stock_shares[stock]} price now: {context[stock + price]}")
                if decision == 'long':
                    '''
                    if decision is long subtract valuation at t from t+1
                    '''
                    tmp = context[stock + price]*stock_shares[stock] - portfolio_values[decision][stock]
                    reward += context[stock + price]*stock_shares[stock] - portfolio_values[decision][stock]

                elif decision == 'short':
                    '''
                    if decision is long subtract valuation at t+1 from t
                    '''
                    tmp = portfolio_values[decision][stock] - context[stock + price]*stock_shares[stock]
                    reward += portfolio_values[decision][stock] - context[stock + price]*stock_shares[stock]
                # reward += abs(context[stock + price]*stock_shares[stock] - portfolio_values[decision][stock])
                print(f"[INFO]          For stock {stock} go {decision} with {stock_shares[stock]} shares PRC_t: { prev_context[stock + price]} PRC_t+1: { context[stock + price]} reward: {tmp}")


        current_total_money = self.previous_total_money + reward
        # print(f"[INFO]      Investment : {current_total_money}")
        print(f"[INFO]      Return : {reward}")
        self.money_pool = current_total_money

        state.current_money_pool = self.money_pool

        return reward

    def reset(self, args=None):
        self.previous_total_money = self.money_pool