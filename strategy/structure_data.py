import pandas as pd
import datetime as dt
import numpy as np
import pickle
import datetime
import matplotlib.pyplot as plt
import matplotlib.style

from dateutil.relativedelta import relativedelta

from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm as svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import LassoCV
from sklearn.calibration import CalibratedClassifierCV

matplotlib.style.use("ggplot")

features = [
    "TOT_REVNU",
    "COST_GOOD_SOLD",
    "GROSS_PROFIT",
    "TOT_DEPREC_AMORT",
    "INT_EXP_OPER",
    "INT_INVST_INCOME_OPER",
    "RES_DEV_EXP",
    "IN_PROC_RES_DEV_EXP_AGGR",
    "TOT_SELL_GEN_ADMIN_EXP",
    "RENTAL_EXP_IND_BROKER",
    "PENSION_POST_RETIRE_EXP",
    "OTHER_OPER_INCOME_EXP",
    "TOT_OPER_EXP",
    "OPER_INCOME",
    "NON_OPER_INT_EXP",
    "INT_CAP",
    "ASSET_WDOWN_IMPAIR_AGGR",
    "RESTRUCT_CHARGE",
    "MERGER_ACQ_INCOME_AGGR",
    "RENTAL_INCOME",
    "SPCL_UNUSUAL_CHARGE",
    "IMPAIR_GOODWILL",
    "LITIG_AGGR",
    "GAIN_LOSS_SALE_ASSET_AGGR",
    "GAIN_LOSS_SALE_INVST_AGGR",
    "STOCK_DIV_SUBSID",
    "INCOME_LOSS_EQUITY_INVST_OTHER",
    "PRE_TAX_MINORITY_INT",
    "INT_INVST_INCOME",
    "OTHER_NON_OPER_INCOME_EXP",
    "TOT_NON_OPER_INCOME_EXP",
    "PRE_TAX_INCOME",
    "TOT_PROVSN_INCOME_TAX",
    "INCOME_AFT_TAX",
    "MINORITY_INT",
    "EQUITY_EARN_SUBSID",
    "INVST_GAIN_LOSS_OTHER",
    "OTHER_INCOME",
    "INCOME_CONT_OPER",
    "INCOME_DISCONT_OPER",
    "INCOME_BEF_EXORD_ACCT_CHANGE",
    "EXORD_INCOME_LOSS",
    "CUMUL_EFF_ACCT_CHANGE",
    "CONSOL_NET_INCOME_LOSS",
    "NON_CTL_INT",
    "NET_INCOME_PARENT_COMP",
    "PREF_STOCK_DIV_OTHER_ADJ",
    "NET_INCOME_LOSS_SHARE_HOLDER",
    "EPS_BASIC_CONT_OPER",
    "EPS_BASIC_DISCONT_OPER",
    "EPS_BASIC_ACCT_CHANGE",
    "EPS_BASIC_EXTRA",
    "EPS_BASIC_CONSOL",
    "EPS_BASIC_PARENT_COMP",
    "BASIC_NET_EPS",
    "EPS_DILUTED_CONT_OPER",
    "EPS_DILUTED_DISCONT_OPER",
    "EPS_DILUTED_ACCT_CHANGE",
    "EPS_DILUTED_EXTRA",
    "EPS_DILUTED_CONSOL",
    "EPS_DILUTED_PARENT_COMP",
    "DILUTED_NET_EPS",
    "DILUTION_FACTOR",
    "AVG_D_SHARES",
    "AVG_B_SHARES",
    "NORM_PRE_TAX_INCOME",
    "NORM_AFT_TAX_INCOME",
    "EBITDA",
    "EBIT",
    "CASH_STERM_INVST",
    "NOTE_LOAN_RCV",
    "RCV_EST_DOUBT",
    "RCV_TOT",
    "INVTY",
    "PREPAID_EXPENSE",
    "DEF_CHARGE_CURR",
    "DEF_TAX_ASSET_CURR",
    "ASSET_DISCONT_OPER_CURR",
    "OTHER_CURR_ASSET",
    "TOT_CURR_ASSET",
    "GROSS_PROP_PLANT_EQUIP",
    "TOT_ACCUM_DEPREC",
    "NET_PROP_PLANT_EQUIP",
    "NET_REAL_ESTATE_MISC_PROP",
    "CAP_SOFTWARE",
    "LTERM_INVST",
    "ADV_DEP",
    "LTERM_RCV",
    "INVTY_LTERM",
    "GOODWILL_INTANG_ASSET_TOT",
    "DEF_CHARGE_NON_CURR",
    "DEF_TAX_ASSET_LTERM",
    "ASSET_DISCONT_OPER_LTERM",
    "PENSION_POST_RETIRE_ASSET",
    "OTHER_LTERM_ASSET",
    "TOT_LTERM_ASSET",
    "TOT_ASSET",
    "NOTE_PAY",
    "ACCT_PAY",
    "DIV_PAY",
    "OTHER_PAY",
    "ACCRUED_EXP",
    "OTHER_ACCRUED_EXP",
    "CURR_PORTION_DEBT",
    "CURR_PORTION_CAP_LEASE",
    "CURR_PORTION_TAX_PAY",
    "DEFER_REVNU_CURR",
    "DEFER_TAX_LIAB_CURR",
    "LIAB_DISCONT_OPER_CURR",
    "OTHER_CURR_LIAB",
    "TOT_CURR_LIAB",
    "TOT_LTERM_DEBT",
    "DEFER_REVNU_NON_CURR",
    "PENSION_POST_RETIRE_LIAB",
    "DEFER_TAX_LIAB_LTERM",
    "MAND_REDEEM_PREF_SEC_SUBSID",
    "PREF_STOCK_LIAB",
    "MIN_INT",
    "LIAB_DISC_OPER_LTERM",
    "OTHER_NON_CURR_LIAB",
    "TOT_LTERM_LIAB",
    "TOT_LIAB",
    "TOT_PREF_STOCK",
    "COMM_STOCK_NET",
    "ADDTL_PAID_IN_CAP",
    "RETAIN_EARN_ACCUM_DEFICIT",
    "EQUITY_EQUIV",
    "TREAS_STOCK",
    "COMPR_INCOME",
    "DEF_COMPSN",
    "OTHER_SHARE_HOLDER_EQUITY",
    "TOT_COMM_EQUITY",
    "TOT_SHARE_HOLDER_EQUITY",
    "TOT_LIAB_SHARE_HOLDER_EQUITY",
    "COMM_SHARES_OUT",
    "PREF_STOCK_SHARES_OUT",
    "TANG_STOCK_HOLDER_EQUITY",
    "NET_INCOME_LOSS",
    "TOT_DEPREC_AMORT_CASH_FLOW",
    "OTHER_NON_CASH_ITEM",
    "TOT_NON_CASH_ITEM",
    "CHANGE_ACCT_RCV",
    "CHANGE_INVTY",
    "CHANGE_ACCT_PAY",
    "CHANGE_ACCT_PAY_ACCRUED_LIAB",
    "CHANGE_INCOME_TAX",
    "CHANGE_ASSET_LIAB",
    "TOT_CHANGE_ASSET_LIAB",
    "OPER_ACTIVITY_OTHER",
    "CASH_FLOW_OPER_ACTIVITY",
    "NET_CHANGE_PROP_PLANT_EQUIP",
    "NET_CHANGE_INTANG_ASSET",
    "NET_ACQ_DIVST",
    "NET_CHANGE_STERM_INVST",
    "NET_CHANGE_LTERM_INVST",
    "NET_CHANGE_INVST_TOT",
    "INVST_ACTIVITY_OTHER",
    "CASH_FLOW_INVST_ACTIVITY",
    "NET_LTERM_DEBT",
    "NET_CURR_DEBT",
    "DEBT_ISSUE_RETIRE_NET_TOT",
    "NET_COMM_EQUITY_ISSUED_REPURCH",
    "NET_PREF_EQUITY_ISSUED_REPURCH",
    "NET_TOT_EQUITY_ISSUED_REPURCH",
    "TOT_COMM_PREF_STOCK_DIV_PAID",
    "FIN_ACTIVITY_OTHER",
    "CASH_FLOW_FIN_ACTIVITY",
    "FGN_EXCHANGE_RATE_ADJ",
    "DISC_OPER_MISC_CASH_FLOW_ADJ",
    "INCR_DECR_CASH",
    "BEG_CASH",
    "END_CASH",
    "STOCK_BASED_COMPSN",
    "COMM_STOCK_DIV_PAID",
    "PREF_STOCK_DIV_PAID",
    "TOT_DEPREC_AMORT_QD",
    "STOCK_BASED_COMPSN_QD",
    "CASH_FLOW_OPER_ACTIVITY_QD",
    "NET_CHANGE_PROP_PLANT_EQUIP_QD",
    "COMM_STOCK_DIV_PAID_QD",
    "PREF_STOCK_DIV_PAID_QD",
    "TOT_COMM_PREF_STOCK_DIV_QD",
    "CURR_RATIO",
    "NON_PERFORM_ASSET_TOT_LOAN",
    "LOAN_LOSS_RESERVE",
    "LTERM_DEBT_CAP",
    "TOT_DEBT_TOT_EQUITY",
    "GROSS_MARGIN",
    "OPER_PROFIT_MARGIN",
    "EBIT_MARGIN",
    "EBITDA_MARGIN",
    "PRETAX_PROFIT_MARGIN",
    "PROFIT_MARGIN",
    "FREE_CASH_FLOW",
    "LOSS_RATIO",
    "EXP_RATIO",
    "COMB_RATIO",
    "ASSET_TURN",
    "INVTY_TURN",
    "RCV_TURN",
    "DAY_SALE_RCV",
    "RET_EQUITY",
    "RET_TANG_EQUITY",
    "RET_ASSET",
    "RET_INVST",
    "FREE_CASH_FLOW_PER_SHARE",
    "BOOK_VAL_PER_SHARE",
    "OPER_CASH_FLOW_PER_SHARE",
    "WAVG_SHARES_OUT",
    "WAVG_SHARES_OUT_DILUTED",
    "EPS_BASIC_NET"
]


def add_month(t):
    return t.replace(day=1) + relativedelta(months=2) - datetime.timedelta(days=1)


def dump_data(df, fn, dump_xls=False):
    pickle._dump(df, open(dump_path + "/" + fn, 'wb'))
    if dump_xls:
        df.to_excel(dump_path + "/" + fn + ".xlsx")


def run_classifier(tr_x, tr_y, test_x, test_y, mdl, name):
    mdl.fit(tr_x, tr_y)
    pred = mdl.predict(test_x)
    success_rate = len(test_y[test_y == pred]) / len(test_y)
    print(name + ": ", success_rate)
    return success_rate

path = "../ZFB-complete"
outpath = "../ZFB-output"
dump_path = '../dump1'

imp_cols = ['TICKER', 'PER_END_DATE', 'PER_TYPE'] + features

dt_cut = datetime.datetime(2010, 1, 1)


def import_data():
    """
    Import i) Stock Fundamental; ii) SP500; iii) Stock Price
    Keep selected Features
    Merge Data
    Calculate Returns and Classcification
    Split Training/Test
    """

    data = pd.DataFrame()
    for i in range(1, 6, 1):
        data = data.append(
            pd.read_csv(
                outpath + '/ZFB-' + str(i) + '-Cleaned.csv',
                usecols=imp_cols
            )
        )

    data = data[data.PER_TYPE == 'Q']

    data.PER_END_DATE = [
        dt.datetime.strptime(x, '%m/%d/%Y').date()
        for x in data.PER_END_DATE
    ]
    data['TRADE_DATE'] = [add_month(t) for t in data.PER_END_DATE]

    tickers = pickle.load(open(path + '/updated_tickers.p', 'rb'))
    data_slt = data[data.TICKER.isin(tickers)]

    data_ticker = data_slt.set_index(['TICKER', 'TRADE_DATE'])
    data_ticker.sort_index(inplace=True)
    data_ticker = data_ticker[['PER_END_DATE'] + features]

    dump_data(data_ticker, "data_ticker_trade", dump_xls=False)

    data_sp = pd.read_csv(
        path + '/SP500.csv', usecols=['Date', 'Adj Close']
    )
    data_sp.sort_values(by='Date', inplace=True)
    data_sp['TRADE_DATE'] = [
        dt.datetime.strptime(x, '%Y-%m-%d').date()
        for x in data_sp.Date
    ]
    data_sp.set_index('TRADE_DATE', inplace=True)
    dump_data(data_sp[['Adj Close']], "data_sp", dump_xls=False)

    data_price = pickle.load(open(path + '/P', 'rb'))

    data_sp = data_sp.asfreq('M', method='pad')

    data_sp['SP_RETURN'] = data_sp['Adj Close'].pct_change(3).shift(-3).values

    data_set = data_ticker.merge(
        data_price,
        left_index=True,
        right_index=True,
        how='inner'
    )

    data_set['RETURN'] = data_set.groupby(level=0)['PRICE'].pct_change().shift(-1)

    data_set = pd.merge(
        data_set.reset_index(),
        data_sp[['SP_RETURN']].reset_index(),
        on=['TRADE_DATE'],
        how='inner'
    ).set_index(['TICKER', 'TRADE_DATE'])

    data_set.sort_index(inplace=True)
    dump_data(data_set, "data_merged", dump_xls=False)

    df_null_count = data_set[features].isnull().sum().to_frame()
    fld_notnull = df_null_count[df_null_count[0] < 2000].index.tolist()

    data_num = data_set[fld_notnull + ['RETURN', 'SP_RETURN']].dropna()
    data_num['EXCESS'] = (data_num['RETURN'] - data_num['SP_RETURN'])

    data_num['PERFORMANCE'] = [
        (lambda x: -1 if x < -0 else 1)(x)
        for x in data_num.RETURN.values
    ]

    #     data_num['PERFORMANCE'] = [
    #         (lambda x: -1 if x < -0 else 1 )(x)
    #         for x in data_num.EXCESS.values
    #     ]

    feature_slt = fld_notnull[1:-3]

    return data_set, data_num, feature_slt


def prep_data(mydata, myfeatures, feature_mdl, th=1e-5):
    mask = mydata.index.get_level_values(1) < dt_cut
    df_training = mydata.loc[mask]

    training_data = df_training[myfeatures]
    scaler = preprocessing.StandardScaler().fit(training_data)
    training_data = scaler.transform(training_data)
    training_labels = df_training['PERFORMANCE']

    clf = feature_mdl.fit(training_data, training_labels)
    slt_mdl = SelectFromModel(clf, prefit=True, threshold=th)

    training_data_new = slt_mdl.transform(training_data)

    print("Training Data Shape Change")
    print("Before Selection: ", training_data.shape)
    print("After Selection:", training_data_new.shape)

    mask = mydata.index.get_level_values(1) >= dt_cut
    df_test = mydata.loc[mask]

    test_data = df_test[myfeatures]
    test_data = scaler.transform(test_data)
    test_data_new = slt_mdl.transform(test_data)
    test_labels = df_test['PERFORMANCE']

    print("Test Data Shape Change")
    print("Before Selection: ", test_data.shape)
    print("After Selection:", test_data_new.shape)

    return training_data_new, training_labels, test_data_new, test_labels, df_training, df_test, slt_mdl


def select_stock(df_test, test_data, slt_features, mdl, myfeatures):
    pred = mdl.predict(test_data)
    df_trade = df_test.copy()
    df_trade[slt_features] = test_data
    df_trade['PERFORMANCE'] = df_test.PERFORMANCE
    df_trade['PREDICT'] = pred
    df_trade['RECOMMENDATION'] = df_trade['PREDICT'].replace(1, 'BUY').replace(-1, 'SELL')

    pred_proba = mdl.predict_proba(test_data)

    df_trade['SELL_PROB'] = [x[0] for x in pred_proba]
    df_trade['BUY_PROB'] = [x[1] for x in pred_proba]

    df_trade.reset_index(inplace=True)
    df_trade.set_index(['TRADE_DATE', 'TICKER'], inplace=True)
    df_trade.sort_index(inplace=True)

    dump_data(df_trade, "df_result", dump_xls=False)

    return df_trade
