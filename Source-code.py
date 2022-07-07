import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
import math
import numpy as np
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from scipy.stats.stats import pearsonr

# ------------------------ QUESTION 1------------------------- #
print("1")

# Reading the file
df = pd.read_csv('daily_covid_cases.csv')
# Generating x-ticks
x = [16]
for i in range(10):
    x.append(x[i] + 60)

# (a)
xlabels = ['Feb-20', 'Apr-20', 'Jun-20', 'Aug-20', 'Oct-20', 'Dec-20', 'Feb-21', 'Apr-21', 'Jun-21', 'Aug-21', 'Oct-21']
new_cases = df['new_cases']
plt.figure(figsize=(20, 10))
plt.xticks(x, xlabels)
plt.xlabel('Month-Year')
plt.ylabel('New Confirmed Cases')
plt.title('Lineplot-Q1a\nCases vs Months')
plt.plot(new_cases)
plt.show()
print('\n(a) Plot displayed')

# (b)
# Generating 1 day lag
lagged = df['new_cases'].shift(1)
corr = pearsonr(lagged[1:], new_cases[1:])
print("\n(b)\nThe required Pearson autocorrelation: ",round(corr[0], 3))


# (c)
plt.xlabel('Given time sequence')
plt.ylabel('One-day lagged generated sequence')
plt.title('Scatterplot-Q1c\nOne-day lagged sequence vs. Given time sequence')
plt.scatter(new_cases[1:], lagged[1:])
plt.show()
print('\n(c) Plot displayed')

# (d)
# Lag values
lag = [1, 2, 3, 4, 5, 6]
correlation = []
print("\n(d)\nThe correlation coefficient between each of the generated time sequences and the given time sequence:")
for d in lag:
    lagged = df['new_cases'].shift(d)
    corr = pearsonr(lagged[d:], new_cases[d:])
    correlation.append(corr[0])
    print(f"{d}-day =", round(corr[0],3))
# line plot of correlation coefficients vs lagged values
plt.xlabel('Lagged Values')
plt.ylabel('Correlation coefficients')
plt.title('Q1d\nCorrelation coefficients vs Lagged Values')
plt.plot(lag, correlation)
plt.show()
print('Plot obtained')

# (e)
sm.graphics.tsa.plot_acf(new_cases,lags=lag)
plt.xlabel('Lagged Values')
plt.ylabel('Correlation coefficients')
plt.show()
print('\n(e) Plot obtained')

# ------------------------ QUESTION 2------------------------- #
print('\n2')
# Train test split
series = pd.read_csv('daily_covid_cases.csv', parse_dates=['Date'],index_col=['Date'],sep=',')
test_size = 0.35 # 35% for testing
X = series.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

# (a)
# Training the model
p = 5
model = AutoReg(train, lags=p)
# Fit/train the model
model_fit = model.fit()
# Get the coefficients of AR model 
coef = model_fit.params 
# Printing the coefficients
print('\n(a)\nThe coefficients obtained from the AR model are', coef)

# (b)
#using these coefficients walk forward over time steps in test, one step each time
history = train[len(train)-p:]
history = [history[i] for i in range(len(history))]
predicted = list() # List to hold the predictions, 1 step at a time for t in range(len(test)):
for t in range(len(test)):
  length = len(history)
  Lag = [history[i] for i in range(length-p,length)] 
  yhat = coef[0] # Initialize to w0
  for d in range(p):
    yhat += coef[d+1] * Lag[p-d-1] # Add other values 
  obs = test[t]
  predicted.append(yhat) #Append predictions to compute RMSE later
  history.append(obs) # Append actual test value to history, to be used in next step.


# (i)
# Scatter plot between actual and predicted values
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Q2b(i)\nActual vs Predicted values')
plt.scatter(predicted, test)
plt.show()
print('\n(b)(i) plot obtained')

# (ii)
# line plot between actual and predicted values
plt.figure(figsize=(20, 10))
plt.xlabel('Days')
plt.ylabel('New Cases')
plt.title('Q2b(ii)\nPredicted and Actual values')
plt.plot(test)
plt.plot(predicted)
plt.show()
print('\n(b)(ii) plot obtained')

# (iii)
print('\n(iii)')
# Rmse
rmse_per = (math.sqrt(mean_squared_error(test, predicted,squared=False))/np.mean(test))*100
print('RMSE(%):',rmse_per)

# MAPE
mape = np.mean(np.abs((test - predicted)/test))*100
print('MAPE:',mape)

# ------------------------ QUESTION 3------------------------- #
print('\n3')
lag_val = [1,5,10,15,25]
RMSE = []
MAPE = []
for l in lag_val:
  model = AutoReg(train, lags=l)
  # Fit/train the model
  model_fit = model.fit()
  coef = model_fit.params 
  history = train[len(train)-l:]
  history = [history[i] for i in range(len(history))]
  predicted = list() # List to hold the predictions, 1 step at a time for t in range(len(test)):
  for t in range(len(test)):
    length = len(history)
    Lag = [history[i] for i in range(length-l,length)] 
    yhat = coef[0] # Initialize to w0
    for d in range(l):
      yhat += coef[d+1] * Lag[l-d-1] # Add other values 
    obs = test[t]
    predicted.append(yhat) #Append predictions to compute RMSE later
    history.append(obs) # Append actual test value to history, to be used in next step.

  # Rmse
  rmse_per = (math.sqrt(mean_squared_error(test, predicted,squared=False))/np.mean(test))*100
  RMSE.append(rmse_per)

  # MAPE
  mape = np.mean(np.abs((test - predicted)/test))*100
  MAPE.append(mape)

# RMSE (%) and MAPE between predicted and original data values wrt lags in time sequence
data = {'Lag value':lag_val,'RMSE(%)':RMSE, 'MAPE' :MAPE}
print('Table 1\n',pd.DataFrame(data))

# plotting RMSE(%) vs. time lag
plt.xlabel('Time Lag')
plt.ylabel('RMSE(%)')
plt.title('Q3\nRMSE(%) vs. time lag')
plt.xticks([1,2,3,4,5],lag_val)
plt.bar([1,2,3,4,5],RMSE)
plt.show()

# plotting MAPE vs. time lag
plt.xlabel('Time Lag')
plt.ylabel('MAPE')
plt.title('Q3\nMAPE vs. time lag')
plt.xticks([1,2,3,4,5],lag_val)
plt.bar([1,2,3,4,5],MAPE)
plt.show()

# ------------------------ QUESTION 4------------------------- #
print('\n4')

p = 1
while p < len(df):
  corr = pearsonr(train[p:].ravel(), train[:len(train)-p].ravel())
  if(abs(corr[0]) <= 2/math.sqrt(len(train[p:]))):
    print('The heuristic value for the optimal number of lags is',p-1)
    break
  p+=1

p=p-1
# training the model
model = AutoReg(train, lags=p)
# fit/train the model
model_fit = model.fit()
coef = model_fit.params 
history = train[len(train)-p:]
history = [history[i] for i in range(len(history))]
predicted = list() # List to hold the predictions, 1 step at a time for t in range(len(test)):
for t in range(len(test)):
  length = len(history)
  Lag = [history[i] for i in range(length-p,length)] 
  yhat = coef[0] # Initialize to w0
  for d in range(p):
    yhat += coef[d+1] * Lag[p-d-1] # Add other values 
  obs = test[t]
  predicted.append(yhat) #Append predictions to compute RMSE later
  history.append(obs) # Append actual test value to history, to be used in next step.

# Rmse
rmse_per = (math.sqrt(mean_squared_error(test, predicted, squared=False))/np.mean(test))*100
print('RMSE(%):',rmse_per)

# MAPE
mape = np.mean(np.abs((test - predicted)/test))*100
print('MAPE:',mape)

# ------------------------------------------------------------ #
