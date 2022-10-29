#Logistic Regression

import pandas as pd
import numpy as np
import math as mt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

df_train = pd.read_csv('../input/ai-academy-logreg/train.csv')
df_test = pd.read_csv('../input/ai-academy-logreg/test.csv')
counter = 0

x_train = df_train.loc[:, ['age', 'job', 'marital', 'education', 'previous', 'duration',
                        'balance', 'housing', 'campaign', 'pdays', 'poutcome']]
x_test = df_test.loc[:, ['age', 'job', 'marital', 'education', 'previous', 'duration',
                        'balance', 'housing', 'campaign', 'pdays', 'poutcome']]
y_train = df_train['y'].astype('category').cat.codes
cat_train = df_train.loc[:, ['job', 'marital', 'education', 'housing', 'poutcome']]
cat_test = df_test.loc[:, ['job', 'marital', 'education', 'housing', 'poutcome']]
for counter in cat_train:
    x_train[counter] = x_train[counter].astype('category').cat.codes
for counter in cat_test:
    x_test[counter] = x_test[counter].astype('category').cat.codes

#---------------------------------------------------------------

train_x, val_x, train_y, val_y = train_test_split(x_train, y_train, train_size=0.6)
xgb = XGBClassifier(gamma = 1.2, subsample = 1.0, max_depth = 7,
                    colsample_bytree = 1.0, n_estimators = 82)
model = LogisticRegression(random_state=0).fit(train_x, train_y)
y_test = model.predict(x_test)
pred_y = model.predict(train_x)
prob_tr = model.predict_proba(train_x)[:, 1]
print('Probability: ', model.score(train_x, train_y))
print('ROC-AUC: ', roc_auc_score(train_y, prob_tr))
pred_val_y = model.predict(val_x)
prob_val = model.predict_proba(val_x)[:,1]
print('Probability: ', model.score(val_x, val_y))
print('ROC-AUC: ', roc_auc_score(val_y, prob_val), '\n\n')

xgb.fit(train_x, train_y)
pred_y2 = xgb.predict(train_x)
prob_tr2 = xgb.predict_proba(train_x)[:, 1]
print('Probability: ', xgb.score(train_x, train_y))
print('ROC-AUC: ', roc_auc_score(train_y, prob_tr2))
pred_val_y2 = xgb.predict(val_x)
prob_val2 = xgb.predict_proba(val_x)[:,1]
print('Probability: ', xgb.score(val_x, val_y))
print('ROC-AUC: ', roc_auc_score(val_y, prob_val2))
y_test = xgb.predict(x_test)
df_test['y'] = y_test
