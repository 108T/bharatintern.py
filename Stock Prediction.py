from sklearn.linear_model import LogisticRegression 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Load data from a CSV file
df = pd.read_csv(r"C:\Users\smily\Downloads\nse.csv")
df.head()

df.shape

df.info()

df.describe()

df.dtypes

df=df.reset_index()['Close']
df

df.isnull().sum()


df


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df=scaler.fit_transform(np.array(df).reshape(-1,1))


print(df)


training_size=int(len(df)*0.75)
test_size=int(len(df))-training_size
train_data, test_data = df[0:training_size,:],df[training_size:len(df),:1]

training_size,test_size

train_data,test_data

def create_feartures(dataset,time_steps=1):
    dataX, dataY =[], []
    for i in range(len(dataset)-time_steps-1):
        a=dataset[i:(i+time_steps),0]
        dataX.append(a)
        dataY.append(dataset[i+time_steps, 0])
    return np.array(dataX), np.array(dataY)



ts=100
x_train, y_train=create_feartures(train_data, ts)
x_test, y_test = create_feartures(test_data,ts)



print(x_train.shape), print(y_train.shape)

model = LinearRegression()
model.fit(x_train, y_train)


y_pred = model.predict(x_test)


from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')


import matplotlib.pyplot as plt
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted (Training)")
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted (Testing)")
plt.tight_layout()
plt.show()


len(test_data)


x_input=test_data[209:].reshape(1,-1)
x_input.shape


temp_input=list(x_input)
temp_input=temp_input[0].tolist()
temp_input


day_new=np.imag(1)
day_pred=np.imag(101)


len(df)


df1=df.tolist()
df1=scaler.inverse_transform(df1).tolist()
plt.plot(df1)
plt.plot(df)







