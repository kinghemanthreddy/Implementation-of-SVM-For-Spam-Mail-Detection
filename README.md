# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages.

2. Read the given csv file and display the few contents of the data.

3. Assign the features for x and y respectively.

4. Split the x and y sets into train and test sets.

5. Convert the Alphabetical data to numeric using CountVectorizer.

6. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

7. Find the accuracy of the model.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: HEMANTH KUMAR B
RegisterNumber:  21222044047
*/
import chardet
file = "/content/spam.csv"
with open(file,'rb') as rawdata:
	result = chardet.detect(rawdata.read(10000))
result
import pandas as pd
dataset = pd.read_csv("/content/spam.csv",encoding="windows-1252")
dataset.head()
dataset.info()
dataset.isnull().sum()
x=dataset["v1"].values
y=dataset["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer() 
x_train=cv.fit_transform(x_train) 
x_test=cv.transform(x_test) 
from sklearn.svm import SVC 
svc=SVC() 
svc.fit(x_train,y_train) 
y_pred=svc.predict(x_test) 
y_pred
from sklearn import metrics 
accuracy=metrics.accuracy_score(y_test,y_pred) 
accuracy

```

## Output:

![image](https://user-images.githubusercontent.com/116530537/204096788-a6836895-1494-44b2-b502-2ad2fa4c1112.png)

![image](https://user-images.githubusercontent.com/116530537/204096800-f57899b7-52af-47ea-8884-091f5e6d2126.png)

![image](https://user-images.githubusercontent.com/116530537/204096814-76766dad-3c90-4911-ba6a-3f8594701e59.png)

![image](https://user-images.githubusercontent.com/116530537/204096832-648668ec-2131-403c-947f-6ad52a55524c.png)

![image](https://user-images.githubusercontent.com/116530537/204096848-a301a05b-0b5b-445d-a414-bd154e84b9bc.png)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
