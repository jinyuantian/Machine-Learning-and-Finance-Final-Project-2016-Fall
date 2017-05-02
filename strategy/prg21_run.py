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

import structure_data

#
# Import Data
#
data_full, mydata, myfeatures = structure_data.import_data()
print(mydata.shape)

#
# Select Features
#
feature_mdl = LinearSVC(C=0.01, penalty="l1", dual=False)

trnx, trny, tstx, tsty, df_train, df_test, slt_mdl= structure_data.prep_data(
    mydata, myfeatures, feature_mdl, th=1e-5
)

slt_features = []
flg_selected = slt_mdl.get_support()

for i in range(len(flg_selected)):
    if flg_selected[i]:
        slt_features.append(myfeatures[i])

print("Selected Features")
print(slt_features)

#
# Build Models
#
mdl_dt = DecisionTreeClassifier()
structure_data.run_classifier(
    trnx, trny,
    tstx, tsty,
    mdl_dt, "Decision Tree"
)

mdl_ada = AdaBoostClassifier(n_estimators=200)
structure_data.run_classifier(
    trnx, trny,
    tstx, tsty,
    mdl_ada, "AdaBoost"
)

mdl_lr = LogisticRegression(C=1., solver='lbfgs')
structure_data.run_classifier(
    trnx, trny,
    tstx, tsty,
    mdl_lr, "Logistic Regression"
)

mdl_svm = svm.LinearSVC(C=2)
structure_data.run_classifier(
    trnx, trny,
    tstx, tsty,
    mdl_svm, "Linear SVM"
)

mdl_calsvm =CalibratedClassifierCV(mdl_svm, cv=2, method='isotonic')
structure_data.run_classifier(
    trnx, trny,
    tstx, tsty,
    mdl_calsvm, "Calibrated Linear SVM"
)

CalibratedClassifierCV(mdl_svm, cv=2, method='isotonic')

mdl_rf = RandomForestClassifier()
structure_data.run_classifier(
    trnx, trny,
    tstx, tsty,
    mdl_rf, "Random Forest"
)

mdl_gb = GradientBoostingClassifier(max_depth=6,n_estimators=50,subsample=0.75)
structure_data.run_classifier(
    trnx, trny,
    tstx, tsty,
    mdl_gb, "GradientBoost"
)

mdl_vote = VotingClassifier(
    estimators=[
        ('dt',  mdl_dt),
        ('ada', mdl_ada),
        ('svm', mdl_svm),
        ('rf',  mdl_rf),
        ('gb',  mdl_gb),
        ('csvm',  mdl_calsvm),
        ('lr',  mdl_lr)
        ]
    )

structure_data.run_classifier(
    trnx, trny,
    tstx, tsty,
    mdl_vote, "Voting Classifier"
)

df_trade = structure_data.select_stock(
    df_test[slt_features+['PERFORMANCE', 'RETURN', 'SP_RETURN']],
    tstx, slt_features, mdl_calsvm, myfeatures
)

print(df_trade.groupby(['PREDICT','PERFORMANCE']).count()['RECOMMENDATION'])


