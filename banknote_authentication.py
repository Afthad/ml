#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('data_banknote_authentication.csv',header=None)
x = dataset.iloc[:,[0,1,2,3]].values
y = dataset.iloc[:,-1].values

#spliting dataset into train set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#Training logistic regression on Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train,y_train)

#predicting result of logistic regression
y_pred = classifier.predict(x_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

#predicting acuracy score
from sklearn.metrics import accuracy_score
print("Accuracy score:",accuracy_score(y_test,y_pred))

#Precision score
from sklearn.metrics import precision_score
print("Precision score: ",precision_score(y_test,y_pred))

#F1 score
from sklearn.metrics import f1_score
print("F1 score:",f1_score(y_test,y_pred))


