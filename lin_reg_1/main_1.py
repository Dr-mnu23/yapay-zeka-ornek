import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#read data
dataframe = pd.read_fwf('brain_body.txt')
x_val = dataframe[['Brain']]
y_val = dataframe[['Body']]

x_val.sort_values(by=['Brain'], inplace=True)
y_val.sort_values(by=['Body'], inplace=True)
x_values= x_val[:50]
y_values= y_val[:50]
print (x_values)
print (y_values)

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()
