## Importing the required libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss,roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pickle
from sklearn.model_selection import GridSearchCV

###############################################################################
###### PART I
# Loading the data and Data Exploration
features = pd.read_csv('features_train.csv',header=None)
labels  = pd.read_csv('labels_train.csv',header=None)
dataset = features.copy()
dataset[25] = labels # column 25 is the labels 
dataset = dataset.dropna(axis = 0)
print('dataset shape : ', dataset.shape)
print('\n')
print(dataset.head())
print('\n')
print(dataset.info())
print('\n')
print(dataset.describe())

# Assuming label zero as cancer negative cases and label 1 as cancer positive cases
a, b = dataset[25].value_counts()
print('\nNumber of patients tested positive with cancer: ', a)
print('Number of people tested negative cancer: ', b)

## visualising the number of positve and negative cases
sns.set(rc={"figure.figsize":(4, 4)}) #width=4, #height=4
sns.countplot(data=dataset, x = 25, label='Count')
plt.xlabel('Label')
plt.show()

## plotting the correlation matrix

correlations = dataset.corr(method='pearson')
sns.set(rc={"figure.figsize":(20, 20)}) #width=20, #height=20
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .75})
plt.show()

# Calculating the Variance inflating factor to test multicollinearity
vif = pd.DataFrame()
vif["Features"] = features.columns
vif["VIF"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]

print(vif)

## Visualising different VIF values using bar plot
sns.set(rc={"figure.figsize":(8, 8)}) #width=8, #height=8
plt.xlabel('VIF')
plt.title('Classifier ROC_AUC score')

sns.set_color_codes("muted")
p = sns.barplot(x=vif['Features'] , y=vif['VIF'], data=vif, color="salmon")
p.set_ylim(0.0, 4.0)
plt.show()

'''
################# TIME CONSUMING PLOT #########################
## Plotting pair plots to understand each features
sns.set(rc={"figure.figsize":(20, 20)}) #width=20, #height=20
sns.pairplot(dataset,hue = 25)
plt.show()
###############################################################
'''

## Plotting KDE plots to understand each features
sns.set(rc={"figure.figsize":(20, 20)}) #width=3, #height=4
fig, axes = plt.subplots(5, 5)
for i in range(25):
  j = i//5
  k = i%5
  sns.kdeplot(dataset.loc[(dataset[25]==1),
            i], color='r', shade=True, label='cancer positive',ax=axes[j,k])
  sns.kdeplot(dataset.loc[(dataset[25]==0),
            i], color='g', shade=True, label='cancer negative',ax=axes[j,k])

plt.show()

###############################################################################
##### PART II

# Modelling 

#spliting the data in to test and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size = 0.1)
y_train = np.ravel(y_train)
y_valid = np.ravel(y_valid)

# Fitting and XGBoost model
xgb = XGBClassifier(n_estimators=70, max_depth= 2, objective='binary:logistic', eval_metric = 'auc')
#model = XGBClassifier()
xgb.fit(X_train, y_train)

#predicting the values of the validation set
predict = xgb.predict(X_valid)

## Assess trained model performance on Training set
sns.set(rc={"figure.figsize":(4, 4)}) #width=4, #height=4
predict_train = xgb.predict(X_train)
cm = confusion_matrix(y_train, predict_train)
plt.figure()
plt.title('Confusion matrix from Training data')
sns.heatmap(cm, annot=True)
plt.show()

# print metrics for Training set
print("\nPrecision_train = {}".format(precision_score(y_train, predict_train)))
print("Recall_train = {}".format(recall_score(y_train, predict_train)))
print("Accuracy_train = {}".format(accuracy_score(y_train, predict_train)))

print("\nPrecision_valid = {}".format(precision_score(y_valid, predict)))
print("Recall_valid = {}".format(recall_score(y_valid, predict)))
print("Accuracy_valid = {}\n".format(accuracy_score(y_valid, predict)))

## plot the confusion matrix for validation set
sns.set(rc={"figure.figsize":(4, 4)}) #width=4, #height=4
cm = confusion_matrix(y_valid, predict)
plt.figure()
plt.title('Confusion matrix from validation data')
sns.heatmap(cm, annot=True)
plt.show()

#predicting the probabilites
xgb.predict_proba(X_valid)[:,1]

# Estimating the auc score
auc = roc_auc_score(y_valid,xgb.predict_proba(X_valid)[:,1] )
print("Auc for our validation data is {}\n". format(auc))

## Plotting the auc curve
fpr, tpr, thresholds = metrics.roc_curve(y_valid, xgb.predict_proba(X_valid)[:,1])
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                   estimator_name='XGBoost estimator')
display.plot()
plt.show()

#saving the model 
#pickle.dump(xgb, open("xgb.pickle.dat", "wb"))

'''
loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
# make predictions for test data
y_pred = loaded_model.predict(X_valid) 

'''

# Hyper parameter tuning
param_grid = {
        'gamma': [0.25, 0.35, 0.15, 0.75],
        'n_estimators': [25,50,10,35,70],
        'max_depth': [2,3,4,5]
        }

xgb_model = XGBClassifier(objective='binary:logistic',eval_metric = 'auc')
grid = GridSearchCV(xgb_model, param_grid, verbose = 4)
grid.fit(X_train, y_train)

print(grid.best_estimator_)

y_predict_optim = grid.predict(X_valid)

print("Precision = {}".format(precision_score(y_valid, y_predict_optim)))
print("Recall = {}".format(recall_score(y_valid, y_predict_optim)))
print("Accuracy = {}".format(accuracy_score(y_valid, y_predict_optim)))

###############################################################################
###### PART III

# Other Methods


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss,roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
classifiers = [
    xgb,
    KNeighborsClassifier(5),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(max_iter = 5000)]

log_cols = ["Classifier", "ROC_AUC_score"]
log      = pd.DataFrame(columns=log_cols)

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

 

auc_dict = {}
    
for clf in classifiers:
    name = clf.__class__.__name__
    clf.fit(X_train, y_train)
    train_predictions = clf.predict_proba(X_valid)[:,1]
    auc = roc_auc_score(y_valid, train_predictions)
    if name in auc_dict:
        auc_dict[name] += auc
    else:
        auc_dict[name] = auc

for clf in auc_dict:
    log_entry = pd.DataFrame([[clf, auc_dict[clf]]], columns=log_cols)
    log = log.append(log_entry)

plt.xlabel('ROC AUC score')
plt.title('Classifier ROC_AUC score on Validation set')

sns.set_color_codes("muted")
sns.barplot(x='ROC_AUC_score', y='Classifier', data=log, color="b")
plt.show()


################### Ensembling ################################################
AdaBoost = AdaBoostClassifier()
GradientBoost = GradientBoostingClassifier()
LDA = LinearDiscriminantAnalysis()
QDA = QuadraticDiscriminantAnalysis()
LG = LogisticRegression(max_iter = 5000)

selected_classifiers = [
    xgb,
    AdaBoost,
    GradientBoost,
    LDA,
    QDA,
    LG]
    
pred_dict = {}    
for clf in selected_classifiers:
    name = clf.__class__.__name__
    clf.fit(X_train, y_train)
    valid_predictions = clf.predict_proba(X_valid)[:,1]
    pred_dict[name]=valid_predictions
    


ensemble_prediction = 0
for clf in selected_classifiers:
    name = clf.__class__.__name__
    ensemble_prediction += pred_dict[name]
    
ensemble_prediction = ensemble_prediction/6


## Plotting the auc curve
fpr, tpr, thresholds = metrics.roc_curve(y_valid, ensemble_prediction)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                   estimator_name='ensemble estimator')
display.plot()
plt.show()

## Assess trained model performance on Training set
sns.set(rc={"figure.figsize":(4, 4)}) #width=4, #height=4
cm = confusion_matrix(y_valid, np.round(ensemble_prediction))
plt.figure()
plt.title('Confusion matrix from ensemble prediction')
sns.heatmap(cm, annot=True)
plt.show()


# Finally Training XGBOOST on the whole training data
xgb.fit(features, labels)

predict_train_final = xgb.predict(features)
print("\nPrecision_train_final = {}".format(precision_score(labels, predict_train_final)))
print("Recall_train_final = {}".format(recall_score(labels, predict_train_final)))
print("Accuracy_train_final = {}".format(accuracy_score(labels,predict_train_final)))


#pickle.dump(xgb, open("xgb.pickle.dat", "wb"))
