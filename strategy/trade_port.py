import pandas as pd
import datetime as dt
import numpy as np
import pickle
import datetime
import matplotlib.pyplot as plt
import matplotlib.style
matplotlib.style.use("ggplot")


AMT_INVEST = 100

def sd(port):
    return np.std((np.diff(np.log(port))))


def sr(port):
    return np.average(np.diff(np.log(port))) / sd(port)


def port_summary(p1, p2, p3, port_sp, fn, sumfn, strn, investment=AMT_INVEST):
    port = [sum(x) for x in zip(p1, p2, p3)]
    port_rtn = port[-1] / (3.0 * investment)
    sp_rtn = port_sp[-1] * 1.0 / investment

    lst_rtn = [p1[-1] * 1.0 / investment, p2[-1] * 1.0 / investment, p3[-1] * 1.0 / investment,
               port_rtn, sp_rtn]
    lst_sr = [sr(p1), sr(p2), sr(p3), sr(port), sr(port_sp)]
    lst_sd = [sd(p1), sd(p2), sd(p3), sd(port), sd(port_sp)]
    df_summary = pd.DataFrame(
        {"Return": lst_rtn, "STD": lst_sd, "Sharpe Ratio": lst_sr},
        index=["Portfolio1", "Portfolio2", "Portfolio3", "Total", "SP500"])

    print(df_summary)
    df_summary.to_excel(sumfn)

    fig = plt.figure()

    plt.plot(port_sp, label="SP500", linewidth=5)
    plt.plot(p1, label="Portfolio 1", linewidth=5)
    plt.plot(p2, label="Portfolio 2", linewidth=5)
    plt.plot(p3, label="Portfolio 3", linewidth=5)
    plt.legend(loc=2, fontsize='xx-large')
    plt.title("Returns from Strategy " + strn)
    fig.savefig(fn, dpi=fig.dpi)
    plt.close(fig)


def trade(df_trade, port_mgmt, imgfn, sumfn, strn):
    investment = AMT_INVEST
    port_sp = [investment]
    port_dates = pd.date_range('1/31/2010', periods=24, freq='3m')

    for t in port_dates:
        investment = investment * (1 + df_trade.loc[t].SP_RETURN[0])
        port_sp.append(investment)

    port1 = port_mgmt(df_trade, '1/31/2010')
    port2 = port_mgmt(df_trade, '2/28/2010')
    port3 = port_mgmt(df_trade, '3/31/2010')

    port_summary(port1, port2, port3, port_sp, imgfn, sumfn, strn, investment=100)