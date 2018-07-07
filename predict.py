import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
#reading data into dataframe
df=pd.read_csv("sphist.csv")
#converting Date column to pandas date type
df["Date"]=pd.to_datetime(df["Date"])
#sorting df w.r.t Date column
df=df.sort_values("Date")
#Computing indicators
#Computing mean volume for past 5 days
df["vol_5"]=df["Volume"].rolling(5).mean()
#Computing mean volume for past 365 days
df["vol_365"]=df["Volume"].rolling(365).mean()
#column indicating the mean of past 5 stock price
df["day_5"]=df["Close"].rolling(5).mean()
#column indicating the mean of past 30 stock price
df["day_30"]=df["Close"].rolling(30).mean()
#column indicating the mean of past 365 stock price
df["day_365"]=df["Close"].rolling(365).mean()
#shifting by 1 down as rolling includes the current value
df[["vol_5","vol_365","day_5","day_30","day_365"]]=df[["vol_5","vol_365","day_5","day_30","day_365"]].shift(1)
#removing first 365 rows because they contain Nan
df=df[df["Date"] > datetime(year=1951, month=1, day=2)]
#dropping rows with na values
df=df.dropna(axis=0)
#generating train and test dataframes
train=df[df["Date"]< datetime(year=2013, month=1, day=1)]
test=df[df["Date"]> datetime(year=2013, month=1, day=1)]
#Predicting stock prices 
lr=LinearRegression()
lr.fit(train[["day_5"]],train["Close"])
predictions1=lr.predict(test[["day_5"]])
mae1=mean_absolute_error(predictions1,test["Close"])
lr.fit(train[["day_30"]],train["Close"])
predictions2=lr.predict(test[["day_30"]])
mae2=mean_absolute_error(predictions2,test["Close"])
lr.fit(train[["day_365"]],train["Close"])
predictions3=lr.predict(test[["day_365"]])
mae3=mean_absolute_error(predictions3,test["Close"])
lr.fit(train[["day_5","day_30"]],train["Close"])
predictions4=lr.predict(test[["day_5","day_30"]])
mae4=mean_absolute_error(predictions4,test["Close"])
lr.fit(train[["day_30","day_365"]],train["Close"])
predictions5=lr.predict(test[["day_30","day_365"]])
mae5=mean_absolute_error(predictions5,test["Close"])
lr.fit(train[["day_5","day_30","day_365"]],train["Close"])
predictions6=lr.predict(test[["day_5","day_30","day_365"]])
mae6=mean_absolute_error(predictions6,test["Close"])
lr.fit(train[["vol_5"]],train["Close"])
predictions7=lr.predict(test[["vol_5"]])
mae7=mean_absolute_error(predictions7,test["Close"])
lr.fit(train[["vol_365"]],train["Close"])
predictions8=lr.predict(test[["vol_365"]])
mae8=mean_absolute_error(predictions8,test["Close"])
print(mae1)
print(mae2)
print(mae3)
print(mae4)
print(mae5)
print(mae6) 
print(mae7)
print(mae8)