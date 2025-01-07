'''
Portfolio class to store stock valuation info against decisions
'''
class Portfolio:
    def __init__(self, current_money_pool, context):
        '''
        current money pool: Stor e the money pool for the iteration
        context: info regarding stock on a particular date
        '''
        self.shares = {}
        self.current_money_pool = current_money_pool
        self.calc_context = context

    def add_stock(self, decision: str, stock: str, purchased_shares: int):
        self.shares[decision][stock] = purchased_shares

    def add_stock(self, purchased_shares: dict):
        '''
        Store entire information for decision, stock and amouht of stock purchased
        '''
        self.shares = purchased_shares
        # for decision in purchased_shares.keys():
        #     for stock, shares in purchased_shares[decision]:
        #         self.shares[decision][stock] += shares

    def calculate_value(self, context: dict):
        '''
        Calculates total valuation of stock portfolio based on context
        '''
        value = {
            "long":{},
            "short":{}
        }
        shares = {}
        long_shares = self.shares['long']
        for stock, purchased_shares in long_shares.items():
            price = self.calc_context[stock + "_PRC"]
            value['long'][stock] = price * purchased_shares
            shares[stock] = purchased_shares
    
        short_shares = self.shares['short']
        for stock, purchased_shares in short_shares.items():
            price = self.calc_context[stock + "_PRC"]
            value['short'][stock] = price * purchased_shares
            shares[stock] = purchased_shares

        return value, shares, self.calc_context
