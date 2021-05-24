
# XGBoost model

import xgboost as xgb
Xgbr = xgb.XGBRegressor(scale_pos_weight = sum(negative instances) / sum(positive instances))
xgbr.fit(x_train,y_train)
Xgbr.feature_importances_ #看feature importance

# tuning parameter

from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3,4,5],
    'alpha': [0, 0.1, 0.2],
    'lambda': [1.1, 1, 0.9]
}
clf = GridSearchCV(xgbr, param_grid, 
                   scoring='roc_auc',
                   verbose=2)
                   
###########################################################################################################
import pandas as pd
import numpy as np
import seaborn as sn
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost as xgb
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids

import pickle
with open('df_null_50.pickle', 'rb') as f:
    df = pickle.load(f)
    
no_use_var = ['snap_yyyymm', 'hash_customer_id', 'y']
categorical_var = ['profile1', 'profile2', 'profile3', 'profile5', 'profile6']
numerical_var = set(df.columns) - set(categorical_var) - set(no_use_var)
numerical_var = list(numerical_var)

y = df['y']
#x_column = list(set(df.columns) - set(no_use_var))
#x = df[x_column]
x_numerical = x[numerical_var]

x_categorical = x[categorical_var]
x_categorical_dummies = pd.get_dummies(x_categorical)
x = pd.concat([x_numerical, x_categorical_dummies], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=True, stratify = y)

categorical_after = list(set(x_train.columns) - set(numerical_var))

x_train_numerical = x_train[numerical_var]
x_test_numerical = x_test[numerical_var]
x_train_categorical = x_train[categorical_after]
x_test_categorical = x_test[categorical_after]

N = 100

'''
步驟
1. 先填mean
2. sort
3. 選出絕對值最高前N筆
'''
x_train_numerical_mean = x_train_numerical.fillna(x_train_numerical.mean())
corr = x_train_numerical_mean.corrwith(y_train)
corr_sorted = abs(corr).sort_values(ascending = False)
topNfeature = corr_sorted.keys()[1:N]

x_train_numerical_selected = x_train[topNfeature]
x_test_numerical_selected = x_test[topNfeature]

x_train = pd.concat([x_train_numerical_selected, x_train_categorical], axis = 1)
x_test = pd.concat([x_test_numerical_selected, x_test_categorical], axis = 1)

#fill na with 0
x_train = x_train.fillna(0)

#fill na with mode
fill_mode = lambda col : col.fillna(col.mode()[0])
x_train = x_train.apply(fill_mode, axis = 0)

#oversampling
oversample = SMOTE()
x_train, y_train = oversample.fit_resample(x_train, y_train)

#undersampling
undersample = ClusterCentroids(random_state=5487)
x_train, y_train = undersample.fit_resample(x_train, y_train)

### Metrics 
y_score = dt.predict_proba(X_test)[:,1]
fpr,tpr,threshold = metrics.roc_curve(y_test.map({'no':0,'yes':1}), y_score)
roc_auc = metrics.auc(fpr,tpr)
plt.stackplot(fpr, tpr, color='steelblue', alpha = 0.5, edgecolor = 'black')
plt.plot(fpr, tpr, color='black', lw = 1)
plt.plot([0,1],[0,1], color = 'red', linestyle = '--')
plt.text(0.5,0.3,'ROC curve (area = %0.3f)' % roc_auc)
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.show()

###

>>> from sklearn.datasets import load_breast_cancer
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.metrics import roc_auc_score
>>> X, y = load_breast_cancer(return_X_y=True)
>>> clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
>>> roc_auc_score(y, clf.predict_proba(X)[:, 1])
0.99...
>>> roc_auc_score(y, clf.decision_function(X))
0.99...

###

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc  ###計算roc和auc
from sklearn import cross_validation
# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target
##變為2分類
X, y = X[y != 2], y[y != 2]
# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
# shuffle and split training and test sets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=.3,random_state=0)
# Learn to predict each class against the other
svm = svm.SVC(kernel='linear', probability=True,random_state=random_state)
###通過decision_function()計算得到的y_score的值，用在roc_curve()函式中
y_score = svm.fit(X_train, y_train).decision_function(X_test)
# Compute ROC curve and ROC area for each class
fpr,tpr,threshold = roc_curve(y_test, y_score) ###計算真正率和假正率
roc_auc = auc(fpr,tpr) ###計算auc的值
plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率為橫座標，真正率為縱座標做曲線
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

## PR curve
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
      
## PR curve graph1
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

disp = plot_precision_recall_curve(classifier, X_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))

###########################################################################################
precision, recall, _ = precision_recall_curve(y_test, predictions)
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot() 

>>> features = iris['feature_names']
>>> importances = rnd_clf.feature_importances_
>>> indices = np.argsort(importances)

>>> plt.title('Feature Importances')
>>> plt.barh(range(len(indices)), importances[indices], color='b', align='center')
>>> plt.yticks(range(len(indices)), [features[i] for i in indices])
>>> plt.xlabel('Relative Importance')
>>> plt.show()
###########################################################################################
precision, recall, f1, _ = precision_recall_fscore_support(test_y, predicted, 
                                                          average='weighted')

###########################################################################################
0514 電腦17
import pandas as pd
import numpy as np
import seaborn as sn
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, precision_score
from sklearn.metrics import roc_auc_score, PrecisionRecallDisplay, precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score
import xgboost as xgb
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids

#讀資料
with open('df_null_50.pickle', 'rb') as f:
    df = pickle.load(f)

#preprocess
fill_mode = lambda col : col.fillna(col.mode()[0])
def preprocessing_data(df, FillnaWithMode = False, FillnaWith0 = False, isOversampling = False, isUndersampling = False, num_feature_keep = 20):
    no_use_var = ['snap_yyyymm', 'hash_customer_id', 'y']
    categorical_var = ['profile1', 'profile2', 'profile3', 'profile5', 'profile6']
    numerical_var = set(df.columns) - set(categorical_var) - set(no_use_var)
    numerical_var = list(numerical_var)
    y = df['y']
    x_column = list(set(df.columns) - set(no_use_var))
    x = df[x_column]
    x_numerical = x[numerical_var]
    x_categorical = x[categorical_var]
    x_categorical_dummies = pd.get_dummies(x_categorical)
    x = pd.concat([x_numerical, x_categorical_dummies], axis = 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=True, stratify = y)
    categorical_after = list(set(x_train.columns) - set(numerical_var))
    x_train_numerical = x_train[numerical_var]
    x_test_numerical = x_test[numerical_var]
    x_train_categorical = x_train[categorical_after]
    x_test_categorical = x_test[categorical_after]
    
    #select correlation with top N 
    x_train_numerical_mean = x_train_numerical.fillna(x_train_numerical.mean())
    corr = x_train_numerical_mean.corrwith(y_train)
    corr_sorted = abs(corr).sort_values(ascending = False)
    topNfeature = corr_sorted.keys()[1:num_feature_keep]
    x_train_numerical_selected = x_train[topNfeature]
    x_test_numerical_selected = x_test[topNfeature]
    
    x_train = pd.concat([x_train_numerical_selected, x_train_categorical], axis = 1)
    x_test = pd.concat([x_test_numerical_selected, x_test_categorical], axis = 1)
    
    if FillnaWith0:
        x_train = x_train.fillna(0)
    if FillnaWithMode:
        fill_mode = lambda col : col.fillna(col.mode()[0])
        x_train = x_train.apply(fill_mode, axis = 0)
    
    if isOversampling:
        oversample = SMOTE()
        x_train, y_train = oversample.fit_resample(x_train, y_train)
    if isUndersampling:
        undersample = ClusterCentroids(random_state=5487)
        x_train, y_train = undersample.fit_resample(x_train, y_train)
    
    return x_train, y_train, x_test, y_test

#model
negative_instances = y_train.value_counts()[0]
positive_instances = y_train.value_counts()[1]
xgbr = xgb.XGBRegressor(scale_pos_weight = negative_instances / positive_instances)
xgbr.fit(x_train, y_train)
y_pred = xgbr.predict(x_test)
y_pred_bool = np.where(y_pred > 0.5, 1, 0)
def print_ROC_AUC(y_test, y_pred):
    # Compute ROC curve and ROC area for each class
    fpr,tpr,threshold = roc_curve(y_test, y_pred) ###計算真正率和假正率
    roc_auc = auc(fpr,tpr) ###計算auc的值
    print(f'AUC = {np.round(roc_auc, 3)}')
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
    lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率為橫座標，真正率為縱座標做曲線
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def print_precision_recall(y_test, y_pred):
    average_precision = average_precision_score(y_test, y_pred)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.show()

def print_feature_importance(number_show, model):
    feature_importance = model.get_booster().get_score(importance_type = 'weight')
    keys = list(feature_importance.keys())
    values = list(feature_importance.values())
    data = pd.DataFrame(data = values, index = keys, columns = ['score']).sort_values(by = "score", ascending = False)
    print(data.head(number_show))
    data.iloc[:TopNImportantFeature, ].plot(kind = 'barh')
    plt.show()

#跑的code
x_train, y_train, x_test, y_test = preprocessing_data(df, FillnaWithMode = False, FillnaWith0 = True, isOversampling = False, isUndersampling = False, num_feature_keep = 500)
xgbr = xgb.XGBRegressor(scale_pos_weight = negative_instances / positive_instances)
xgbr.fit(x_train, y_train)
y_pred = xgbr.predict(x_test)
y_pred_bool = np.where(y_pred > 0.5, 1, 0)
print_ROC_AUC(y_test, y_pred)
print_precision_recall(y_test, y_pred)
number_show = 20
print_feature_importance(number_show, xgbr)

#random forest
features = x_train.columns()
importances = forest.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
