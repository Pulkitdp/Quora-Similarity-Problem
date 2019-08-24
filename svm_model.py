from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import classification_report as repo
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('new.csv')
x=dataset.iloc[:,:25]
y=dataset.iloc[:,25]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=1)

svc=SVC()
svc.fit(x_train,y_train)
ypred=svc.predict(x_test)

print("Accuracy Score :",acc(ypred,y_test))
print("Classification Report :\n",repo(ypred,y_test))