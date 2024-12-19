class Portfolio:
    def __init__(self, current_money_pool, context):
        self.shares = {}
        self.current_money_pool = current_money_pool
        self.calc_context = context

    def add_stock(self, decision: str, stock: str, purchased_shares: int):
        self.shares[decision][stock] = purchased_shares

    def add_stock(self, purchased_shares: dict):
        self.shares = purchased_shares
        # for decision in purchased_shares.keys():
        #     for stock, shares in purchased_shares[decision]:
        #         self.shares[decision][stock] += shares

    def calculate_value(self, context: dict):
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
