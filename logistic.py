from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import classification_report as repo
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('new.csv')
x=dataset.iloc[:,:25]
y=dataset.iloc[:,25]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=1)

log=LogisticRegression()
log.fit(x_train,y_train)
ypred=log.predict(x_test)

print("Accuracy Score :",acc(ypred,y_test))
print("Classification Report :",repo(ypred,y_test))