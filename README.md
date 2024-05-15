# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM :

To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required :

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## ALGORITHM :

### Step 1 :

Import the standard libraries such as pandas module to read the corresponding csv file.

### Step 2 :

Upload the dataset values and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

### Step 3 :

Import LabelEncoder and encode the corresponding dataset values.

### Step 4 :

Import LogisticRegression from sklearn and apply the model on the dataset using train and test values of x and y.

### Step 5 :

Predict the values of array using the variable y_pred.

### Step 6 :

Calculate the accuracy, confusion and the classification report by importing the required modules such as accuracy_score, confusion_matrix and the classification_report from sklearn.metrics module.

### Step 7 :

Apply new unknown values and print all the acqirred values for accuracy, confusion and the classification report.

Step 8:
End the program.

## Program :

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

### DEVELOPED BY : RAMPRASATH.R
### REGISTER NO : 212223220086

```
import pandas as pd

data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:,:-1]
x
y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output :

### HEAD OF THE DATA :

![Screenshot 2023-08-31 100728](https://github.com/Abrinnisha6/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118889454/94972664-e33a-40ff-8e26-273ef649c017)


### COPY HEAD OF THE DATA :

![Screenshot 2023-08-31 100952](https://github.com/Abrinnisha6/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118889454/8b83173a-35da-4031-974a-4dfafdc9aa7a)

### NULL AND SUM :

![Screenshot 2023-08-31 101034](https://github.com/Abrinnisha6/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118889454/d067060b-a464-4f3d-98f5-ecfa5b41b43c)

### DUPLICATED :

![Screenshot 2023-08-31 101109](https://github.com/Abrinnisha6/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118889454/bbe59d81-7395-4b32-9ae0-92cb2f9f48cd)


### X VALUE :

![Screenshot 2023-08-31 101410](https://github.com/Abrinnisha6/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118889454/009b08db-fb4c-4c6b-a7a0-98a8fa8843ca)

### Y VALUE :

![Screenshot 2023-08-31 101421](https://github.com/Abrinnisha6/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118889454/2f6b359c-aa4c-49ef-9fe9-c67956ab7c71)


### PREDICTED VALUES :

![Screenshot 2023-08-31 101547](https://github.com/Abrinnisha6/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118889454/438504cd-84da-4f8a-a8be-79420fd9ca44)

### ACCURACY :

![Screenshot 2023-08-31 101833](https://github.com/Abrinnisha6/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118889454/f3d03668-13f0-407b-a3db-dfcd1093eac2)

### CONFUSION MATRIX :

![Screenshot 2023-08-31 101858](https://github.com/Abrinnisha6/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118889454/12ab3d4a-4fc5-4fa8-95e4-f3aad3ee8193)

### CLASSIFICATION REPORT :

![Screenshot 2023-08-31 101956](https://github.com/Abrinnisha6/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118889454/d22d2299-5043-457f-9df1-308b42e06d46)


## RESULT :

Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
