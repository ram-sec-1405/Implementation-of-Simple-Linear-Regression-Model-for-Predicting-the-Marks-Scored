# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard Libraries. 
2.Set variables for assigning dataset values. 
3.Import linear regression from sklearn. 4.Assign the points for representing in the graph. 
5.Predict the regression for marks by using the representation of the graph. 
6.Compare the graphs and hence we obtained the linear regression for the given datas.
```

## Program:
```
RAMPRASATH R
212223220086

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
df.head()

![229978451-2b6bdc4f-522e-473e-ae2f-84ec824344c5](https://github.com/Hafeezuldeen/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979314/6f146888-43dc-49e9-b3fc-3105110eab1e)

df.tail()

![229978451-2b6bdc4f-522e-473e-ae2f-84ec824344c5](https://github.com/Hafeezuldeen/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979314/97422886-7852-40f3-8df4-894053f1b742)

Array value of X

![229978918-707c006d-0a30-4833-bf77-edd37e8849bb](https://github.com/Hafeezuldeen/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979314/8075d639-4d77-4f16-94ff-c80ead712420)


Array value of Y
![229978994-b0d2c87c-bef9-4efe-bba2-0bc57d292d20](https://github.com/Hafeezuldeen/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979314/f171eeeb-6c9a-4157-8f17-8e6e972479b2)


Values of Y prediction

![229979053-f32194cb-7ed4-4326-8a39-fe8186079b63](https://github.com/Hafeezuldeen/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979314/edf67094-8606-482c-8252-3c3e9725be2e)


Array values of Y test

![229979114-3667c4b7-7610-4175-9532-5538b83957ac](https://github.com/Hafeezuldeen/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979314/b3cf065e-a3a3-467d-ba98-80f95a024cd7)

Training Set Graph
![229979169-ad4db5b6-e238-4d80-ae5b-405638820d35](https://github.com/Hafeezuldeen/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979314/bf9e5e7e-24af-4f20-a7e3-d617212655d4)

Test Set Graph
![229979225-ba90853c-7fe0-4fb2-8454-a6a0b921bdc1](https://github.com/Hafeezuldeen/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979314/67d16253-2a12-4778-bf2b-9b57967757de)


Values of MSE, MAE and RMSE

![229979276-bb9ffc68-25f8-42fe-9f2a-d7187753aa1c](https://github.com/Hafeezuldeen/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979314/bb2a8916-32b4-46e9-9202-58c354b0ca40)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
