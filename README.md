# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries. 
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph. 
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: BHARATHWAJ R
RegisterNumber:  212222240019
*/
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

`![ML21](https://github.com/BHARATHWAJRAMESH/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394248/89e14bf2-db3b-49fd-ae7e-86e695f1bddf)

df.tail()


![ML22](https://github.com/BHARATHWAJRAMESH/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394248/c89c2790-728d-433e-a9ef-0462a645b098)

Array value of X

![ML23](https://github.com/BHARATHWAJRAMESH/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394248/ede335e6-8525-483b-82af-3d517387c22a)

Array value of Y

![ML24](https://github.com/BHARATHWAJRAMESH/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394248/a612f0ec-7c5d-4240-b262-452f7ed08048)

Values of Y prediction

![ML25](https://github.com/BHARATHWAJRAMESH/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394248/4932d0d3-63d2-4678-af5e-4b4f1503f920)

Array values of Y test

![ML26](https://github.com/BHARATHWAJRAMESH/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394248/5f91fa46-0715-4079-a26b-a3ef920e0a24)

Training Set Graph

![ML27](https://github.com/BHARATHWAJRAMESH/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394248/303f9030-f77f-4006-b1fe-f9fa53f955c0)

Test Set Graph

![ML28](https://github.com/BHARATHWAJRAMESH/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394248/cd8d0dbc-afc4-4ae1-bfbf-cc707927892d)

Values of MSE, MAE and RMSE

![ML29](https://github.com/BHARATHWAJRAMESH/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394248/cd178460-d9a5-4b5a-b88b-560676a43420)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
