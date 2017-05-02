import pickle
import pandas as pd
import numpy as np

import trade_port

path = '../dump1'
imgfn = '../img1/str2.png'

df_trade = pickle.load(open(path + '/df_result', 'rb'))

def port_mgmt(df_trade, dt_bgn, investment=trade_port.AMT_INVEST):
    port_dates = pd.date_range(dt_bgn, periods=24, freq='3m')
    total = [investment]

    for t in port_dates:
        sum_of_probs = sum([2 * x - 1 for x in df_trade.loc[t].BUY_PROB.values])
        returns = df_trade.loc[t]['RETURN'].values
        probs = df_trade.loc[t]['BUY_PROB'].values
        investment = np.dot(
            (investment * (2 * df_trade.loc[t].BUY_PROB.values - 1) / sum_of_probs),
            (1 + returns)
        )
        total.append(investment)

    return total

trade_port.trade(df_trade, port_mgmt, imgfn, path+"/str2.xlsx", "B")
