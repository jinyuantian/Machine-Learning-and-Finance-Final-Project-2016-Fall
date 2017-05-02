import pickle
import pandas as pd
import numpy as np

import trade_port

path = '../dump1'
imgfn = '../img1/str4.png'
df_trade = pickle.load(open(path + '/df_result', 'rb'))

fix_investment = 10

def port_mgmt(df_trade, dt_bgn, investment=trade_port.AMT_INVEST):
    port_dates = pd.date_range(dt_bgn, periods=24, freq='3m')
    total = [investment]

    for t in port_dates:
        buys = df_trade.loc[t].loc[lambda df: df.RECOMMENDATION == 'BUY']
        buys = buys.sort_values(by='BUY_PROB', ascending=False).head(10)

        #         sells = df_trade.loc[t].loc[lambda df: df.RECOMMENDATION == 'SELL']
        #         sells = sells.sort_values(by='SELL_PROB', ascending=False).head(10)

        #         if len(buys) < len(sells):
        #             print(t)
        per_stock = investment / 10

        on_longs = sum([per_stock * (1 + x) for x in buys.RETURN.values])
        #         on_shorts = sum([per_stock*(1+x) for x in sells.RETURN.values])
        investment = on_longs  # - on_shorts

        total.append(investment)

    return total


trade_port.trade(df_trade, port_mgmt, imgfn, path+"/str4.xlsx", "C")
