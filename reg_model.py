from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import classification_report as repo
import matplotlib.pyplot as plt
import pandas as pd
import data
import pickle

# dataset=pd.read_csv('new.csv')
# x=dataset.iloc[:,:25]
# y=dataset.iloc[:,25]

# # x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=1)

# ran=RandomForestClassifier(n_estimators=100,bootstrap=True)
# ran.fit(x,y)

# obj=open('model_saved.pickle','ab')
# pickle.dump(ran,obj)
# obj.close()

file=open('model_saved.pickle','rb')
ran=pickle.load(file)

new_x=data.data_x()
ypred=ran.predict(new_x)
# ypred=ran.predict(x_test)

if ypred == 1:
    print('Questions are similar')
elif ypred == 0:
    print('Questions are not similar')

print("Predction value :", ypred)

# print("Accuracy Score :",acc(ypred,y_test))
# print("Classification Report :\n",repo(ypred,y_test))

# imp=ran.feature_importances_
# col=x.columns
# plt.figure(figsize=(18,8))
# plt.bar(col,imp)
# plt.show()