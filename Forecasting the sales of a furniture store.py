
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#ADF(augmented dickey fuller test)
from statsmodels.tsa.stattools import adfuller,acf,pacf
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
#Arima
from statsmodels.tsa.arima_model import ARIMA
#for lJung box
import statsmodels.api as sm
#load file path
df=pd.read_csv("Super_Store.csv",encoding='latin1')
print(df.columns)
print(df.shape)
#check number of categorical and numerical columns
num_col = [ col for col in df.columns if df[col].dtypes != 'O'] 
print(num_col)
cat_col =[col for col in df.columns if df[col].dtypes =='O']
print(cat_col)
df.info()
#removed all unneccesary columns
df.drop(df.columns.difference(['Order Date','Sales']), 1, inplace=True)
print(df.shape)
print(df.isnull().sum())
#grouped the remaining data
print(df.head())
df = df.groupby("Order Date")['Sales'].sum().reset_index()
print(df.head())
df["Order Date"] = pd.to_datetime(df["Order Date"])
df.set_index("Order Date", inplace = True)
print(df.info())
print(df.index.min() ,df.index.max())
# got montly sales
y = df["Sales"].resample('MS').mean() 
print(y['2015'])
# visualize monthly data
y.plot(figsize=(15,6))
plt.show()
#stationarity check
pvalue = adfuller(y)[1]
if pvalue > 0.05:
     print( 'p-value = {}. Data not stationary'.format(pvalue))
else:
     print('p-value = {}. Data is stationary'.format(pvalue))
   

plot_pacf(y, lags=50)
plt.show()
plot_acf(y, lags=20)
plt.show()
# model 1 -> ARIMA(0,0,0) model
m1 = ARIMA(y,order=(0,0,0)).fit(disp=0)
m1.summary()

# plot the residuals
plt.hist(m1.resid)
plt.title("ARIMA residuals")

# LJungBox test for residuals independence
# H0: residuals are independently distributed
# H1: residuals are not independently distributed
pvalue = sm.stats.acorr_ljungbox(m1.resid,lags=[1])[1]
if pvalue > 0.05:
    print(" Residuals are independently distributed")
else:
    print(" Residuals are not independently distributed")

# forecast for the next 'n' months
p1 = m1.forecast(steps=12)

# forecasted values are in the differenced format
# they have to be converted into the origianl form
predictions = p1[0]
len(predictions)
#print(head(predictions))

