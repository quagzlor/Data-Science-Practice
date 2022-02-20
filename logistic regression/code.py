import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

data = pd.read_csv('Iris.csv')
data = data.drop(['Id'],axis=1)
data = data.drop(data.index[list(range(100,150))])

y = []

target = data['Species']

for flower in target:
    if (flower == 'Iris-setosa'):
        y.append(0)
    else:
        y.append(1)

data = data.drop(['Species'],axis=1)
x = data.values.tolist()

x,y = shuffle(x,y)

x_train = []
y_train = []
x_test = []
y_test = []

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

y_train = y_train.reshape(90,1)

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print(accuracy_score(y_test,y_pred))