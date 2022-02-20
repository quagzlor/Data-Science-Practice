#sci-kit method
import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

x_train = np.array(train['x'])
y_train = np.array(train['y'])

x_test = np.array(test['x'])
y_test = np.array(test['y'])

x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

model = LinearRegression(normalize=True)
model.fit(x_train,y_train)

predict = model.predict(x_test)

print(r2_score(y_test,predict))