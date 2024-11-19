from .oracle import Oracle
from ..portfolio import Portfolio

class PortFolioOracle(Oracle):
    def __init__(self, initial_money_pool = 100.0):
        self.initial_money_pool = initial_money_pool
        self.previous_total_money = self.initial_money_pool

    def calculate_reward(self, state: Portfolio, context: dict):
        current_total_money = state.calculate_value(context)
        reward = current_total_money - self.previous_total_money
        self.previous_total_money = current_total_money

        return reward

    def reset(self, args):
        self.previous_total_money = self.initial_money_pool