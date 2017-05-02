import pickle
import pandas as pd

import trade_port

path = '../dump1'
imgpath = '../img1'
df_trade = pickle.load(open(path + '/df_result', 'rb'))


def port_mgmt(df_trade, dt_bgn, investment=trade_port.AMT_INVEST):
    port_dates = pd.date_range(dt_bgn, periods=24, freq='3m')
    total = [investment]

    for t in port_dates:
        buys = df_trade.loc[t].loc[lambda df: df.RECOMMENDATION == 'BUY']
        sells = df_trade.loc[t].loc[lambda df: df.RECOMMENDATION == 'SELL']
        per_stock = investment / (len(buys) + len(sells))

        on_longs = sum([per_stock * x for x in buys.RETURN.values])
        on_shorts = sum([per_stock * x for x in sells.RETURN.values])
        investment += (on_longs - on_shorts)
        total.append(investment)
        # print(on_longs, on_shorts)

    return total

trade_port.trade(df_trade, port_mgmt, imgpath + "/str1.png", path + "/str1.xlsx", "A")
