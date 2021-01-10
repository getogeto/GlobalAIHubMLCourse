# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 15:56:58 2021

@author: GeTo
"""

# =============================================================================
# Homework 3
# 1) Generate dataset using make_blobs function in the sklearn.datasets class.
# Generate 2000 samples with 3 features (X) with one label (y).
# 2) Explore and analyse raw data.
# 3) Do preprocessing for classification.
# 4) Split your dataset into train and test test (0.7 for train and 0.3 for test).
# 5) Try Decision Tree and XGBoost Algorithm with different hyperparameters. (Using GridSearchCV is a plus)
# 6) Evaluate your result on both train and test set. Analyse if there is any underfitting or
# overfitting problem. Make your comments.
# =============================================================================


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import precision_score,recall_score,accuracy_score,classification_report,f1_score


X, y = make_blobs(n_samples = 2000,
                  n_features = 3,
                  cluster_std = 5.5,
                  random_state = 42)
df = pd.DataFrame(X,columns = ["Group1","Group2","Group3"])
df.head()
df.isnull().sum()
df.dropna()
df.corr()
df.describe()
plt.show(sns.scatterplot(x="Group1",
                         y="Group2",
                         hue=y,
                         data=df))
plt.show(sns.distplot(X.T[0]))
plt.show(sns.distplot(X.T[1]))
plt.show(sns.distplot(X.T[2]))

label_encoder = LabelEncoder()
df["Label"] = label_encoder.fit_transform(df["Group1"])
df.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  

clf_1 = DecisionTreeClassifier(max_depth=4, random_state=42)
clf_1.fit(X_train,y_train)
print("Accuracy of train:",clf_1.score(X_train,y_train))
print("Accuracy of test:",clf_1.score(X_test,y_test))

clf_2 = DecisionTreeClassifier(max_depth=5, random_state=42)
clf_2.fit(X_train,y_train)
print("Accuracy of train:",clf_2.score(X_train,y_train))
print("Accuracy of test:",clf_2.score(X_test,y_test))

dmatrix_train = xgb.DMatrix(data=X_train, label=y_train)
dmatrix_test = xgb.DMatrix(data=X_test, label=y_test)

param = {'max_depth':3, 
         'eta':1, 
         'objective':'multi:softprob', 
         'num_class':3}
num_round = 5
model = xgb.train(param, dmatrix_train, num_round)
pred = model.predict(dmatrix_test)
pred[:10]

pred_0 = np.asarray([np.argmax(line) for line in pred])
print("Precision = {}".format(precision_score(y_test, pred_0, average='macro')))
print("Recall = {}".format(recall_score(y_test, pred_0, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, pred_0)))

param_dict = {"max_depth":range(3,10,2),
              "min_child_weight":range(1,6,2),
              "learning_rate": [0.00001,0.001,0.01,0.1,1,2],
              "n_estimators": [10,190,200,210,500,1000,2000]}

xgc = XGBClassifier(booster = "gbtree",
                    learning_rate = 0.01,
                    n_estimators = 200,
                    max_depth = 5,
                    min_child_weight = 1,
                    gamma = 0,
                    subsample = 0.8,
                    colsample_bytree = 0.8,
                    objective = "multi:softprob",
                    nthread = 4,
                    scale_pos_weight = 1,
                    seed=27)

clf_1 = GridSearchCV(xgc, param_dict, cv=3, n_jobs=-1).fit(X_train,y_train)
print("Tuned: {}".format(clf_1.best_params_)) 
print("Mean of the cv scores is {:.6f}".format(clf_1.best_score_))
print("Train Score {:.6f}".format(clf_1.score(X_train,y_train)))
print("Test Score {:.6f}".format(clf_1.score(X_test,y_test)))
print("Seconds used for refitting the best model on the train dataset: {:.6f}".format(clf_1.refit_time_))
