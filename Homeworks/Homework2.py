# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 21:44:33 2021

@author: GeTo
"""

# =============================================================================
# Homework 2
# Import Boston Dataset from sklearn dataset class.
# Explore and analyse raw data.
# Do preprocessing for regression.
# Split your dataset into train and test test (0.7 for train and 0.3 for test).
# Try Ridge and Lasso Regression models with at least 5 different alpha value for each.
# Evaluate the results of all models and choose the best performing model.
# =============================================================================

import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston

X,y = load_boston(return_X_y=True)

df_boston = pd.DataFrame(X,columns=load_boston().feature_names)
df_boston.head()  
df_boston.info()
df_boston.describe()
df_boston.isna().sum()
df_boston.corr()
corr_boston = df_boston.corr()

plt.figure(figsize=(14,14))
a = sns.heatmap(corr_boston,
                vmin=-1,
                vmax=1,
                center=0,
                cmap=sns.diverging_palette(10,110,n=200),
                square=True,
                annot=True)
a.set_xticklabels(a.get_xticklabels(),
                  rotation=45,
                  horizontalalignment="right")
a.set_ylim(len(corr_boston)+0.5,-0.5)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=42)
b = LinearRegression(normalize=False)
b.fit(X_train,y_train)

print("Score of the train set",b.score(X_train,y_train))
print("Score of the test set",b.score(X_test,y_test))

importance = b.coef_
for i in range(len(importance)):
    print("Feature", df_boston.columns[i], "Score: ", importance[i])

# Dropping colerated features
new_df = df_boston.drop(["AGE","INDUS"],axis=1) 

# Splitting new dataset to 0.7-0.3 train and test
X_train, X_test, y_train, y_test = train_test_split(new_df,  
                                                    y,
                                                    test_size=0.3,
                                                    random_state=42) 
b = LinearRegression(normalize=True)
b.fit(X_train,y_train)
print("Score of the train set",b.score(X_train,y_train))
print("Score of the test set",b.score(X_test,y_test))

# Detecting and eliminating outliers
o = np.abs(stats.zscore(df_boston))
outliers = list(set(np.where(o>3)[0]))
new_df = df_boston.drop(outliers,axis=0).reset_index(drop=False)
new_y = y[list(new_df["index"])]
new_X = new_df.drop("index",axis=1)
X_scaled = StandardScaler().fit_transform(new_X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled,
                                                    new_y,
                                                    test_size=0.3,
                                                    random_state=42)
b = LinearRegression(normalize=False)
b.fit(X_train,y_train)

# Regularized Ridge & Lasso Regressions
ridge_model = Ridge(alpha = 0.02)
ridge_model.fit(X_train, y_train)
lasso_model = Lasso(alpha = 0.001)
lasso_model.fit(X_train, y_train)
print("Simple Train: ", b.score(X_train, y_train))
print("Simple Test: ", b.score(X_test, y_test))
print('---------------------------------------')
print("Lasso Train: ", lasso_model.score(X_train, y_train)) #Lasso
print("Lasso Test: ", lasso_model.score(X_test, y_test))
print('---------------------------------------')
print("Ridge Train: ", ridge_model.score(X_train, y_train)) #Ridge
print("Ridge Test: ", ridge_model.score(X_test, y_test))

ridge_model = Ridge(alpha = 0.2)
ridge_model.fit(X_train, y_train)
lasso_model = Lasso(alpha = 0.1)
lasso_model.fit(X_train, y_train)
print("Simple Train: ", b.score(X_train, y_train))
print("Simple Test: ", b.score(X_test, y_test))
print('---------------------------------------')
print("Lasso Train: ", lasso_model.score(X_train, y_train)) #Lasso
print("Lasso Test: ", lasso_model.score(X_test, y_test))
print('---------------------------------------')
print("Ridge Train: ", ridge_model.score(X_train, y_train)) #Ridge
print("Ridge Test: ", ridge_model.score(X_test, y_test))

ridge_model = Ridge(alpha = 0.002)
ridge_model.fit(X_train, y_train)
lasso_model = Lasso(alpha = 0.5)
lasso_model.fit(X_train, y_train)
print("Simple Train: ", b.score(X_train, y_train))
print("Simple Test: ", b.score(X_test, y_test))
print('---------------------------------------')
print("Lasso Train: ", lasso_model.score(X_train, y_train)) #Lasso
print("Lasso Test: ", lasso_model.score(X_test, y_test))
print('---------------------------------------')
print("Ridge Train: ", ridge_model.score(X_train, y_train)) #Ridge
print("Ridge Test: ", ridge_model.score(X_test, y_test))

ridge_model = Ridge(alpha = 0.003)
ridge_model.fit(X_train, y_train)
lasso_model = Lasso(alpha = 0.07)
lasso_model.fit(X_train, y_train)
print("Simple Train: ", b.score(X_train, y_train))
print("Simple Test: ", b.score(X_test, y_test))
print('---------------------------------------')
print("Lasso Train: ", lasso_model.score(X_train, y_train)) #Lasso
print("Lasso Test: ", lasso_model.score(X_test, y_test))
print('---------------------------------------')
print("Ridge Train: ", ridge_model.score(X_train, y_train)) #Ridge
print("Ridge Test: ", ridge_model.score(X_test, y_test))

ridge_model = Ridge(alpha = 0.9)
ridge_model.fit(X_train, y_train)
lasso_model = Lasso(alpha = 0.9)
lasso_model.fit(X_train, y_train)
print("Simple Train: ", b.score(X_train, y_train))
print("Simple Test: ", b.score(X_test, y_test))
print('---------------------------------------')
print("Lasso Train: ", lasso_model.score(X_train, y_train)) #Lasso
print("Lasso Test: ", lasso_model.score(X_test, y_test))
print('---------------------------------------')
print("Ridge Train: ", ridge_model.score(X_train, y_train)) #Ridge
print("Ridge Test: ", ridge_model.score(X_test, y_test))

## In Lasso Reg with alpha value gets closer to 1, scores decrease.
## Best performing is when alpha is closer to 0 in Ridge Reg.

# =============================================================================
