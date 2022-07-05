import pandas as pd
import pickle
import numpy as np

features = pd.read_csv('features_train.csv',header=None)
labels  = pd.read_csv('labels_train.csv',header=None)
dataset = features.copy()
dataset[25] = labels # column 25 is the labels 
dataset = dataset.dropna(axis = 0)

features_test = pd.read_csv('features_test.csv',header=None)

model = pickle.load(open("xgb.pickle.dat", "rb"))
# make predictions for test data


y_pred = model.predict(features_test)
y_proba = model.predict_proba(features_test)[:,1] 

test_result = pd.DataFrame(y_pred,columns=['Prediction'] )
test_result['Probability'] = y_proba
test_result['Output'] = test_result['Prediction'].map(str) + "," + test_result['Probability'].map(str) 
print(test_result['Output'])

test_result['Output'].to_csv('test_output.csv')
