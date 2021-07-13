import csv
import pandas as pd
from sklearn.linear_model import LogisticRegression

def minMax(x):
    return pd.Series(index=['min','max'],data=[x.min(),x.max()])


Train = 'train.csv'
Test = 'test.csv'
Submission = 'sample_submission.csv'

print("STARTING")

getTrain = pd.read_csv(Train)
getTest = pd.read_csv(Test)
getSubmission = pd.read_csv(Submission)

LabA = 'target_carbon_monoxide'
LabB = 'target_benzene'
LabC = 'target_nitrogen_oxides'

print('trainL: ' , getTrain.shape)
print('test: ' , getTest.shape)
# print(getTrain.apply(minMax))

print(getTrain[LabA])
print(getTrain[LabB])


clfA = LogisticRegression(random_state=0).fit(X, yA)
clfB = LogisticRegression(random_state=0).fit(X, yB)
clfC = LogisticRegression(random_state=0).fit(X, yC)

# print(file)
