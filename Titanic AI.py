import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.interpolate import NearestNDInterpolator
import numpy as np
df = pd.read_csv('/home/ascundar/Downloads/titanic/train.csv')
df2 = pd.read_csv('/home/ascundar/Downloads/titanic/test.csv')
X = df.drop(['Survived', 'PassengerId', 'Cabin', 'Name', 'Ticket'], axis=1)
Test = df2.drop(['PassengerId', 'Cabin', 'Name', 'Ticket'], axis=1)
X['Embarked'] = X['Embarked'].apply(lambda x: 1 if x == 'S' else (2 if x == 'C' else 3))
Test['Embarked'] = Test['Embarked'].apply(lambda x: 1 if x == 'S' else (2 if x == 'C' else 3))
X['Sex'] = X['Sex'].apply(lambda x: 1 if x == 'male' else 5)
Test['Sex'] = Test['Sex'].apply(lambda x: 1 if x == 'male' else 5)
X['Embarked'] = X['Embarked'].apply(lambda x: 1 if x == 'C' else (5 if x == 'Q' else (10 if x == 'S' else 1)))
Test['Embarked'] = Test['Embarked'].apply(lambda x: 1 if x == 'C' else (5 if x == 'Q' else (10 if x == 'S' else 1)))
X['Pclass'] = X['Pclass'].apply(lambda x: 1 if x == '3' else (5 if x == 2 else 10))
Test['Pclass'] = Test['Pclass'].apply(lambda x: 1 if x == '3' else (5 if x == 2 else 10))
y = df['Survived']
X_mod = X.dropna(axis = 0)
Test_mod = Test.dropna(axis = 0)
interp = NearestNDInterpolator(list(zip(X_mod['Pclass'], X_mod['Sex'], X_mod['Fare'], X_mod['Parch'], X_mod['SibSp'])), X_mod['Age'])
interp2 = NearestNDInterpolator(list(zip(Test_mod['Pclass'], Test_mod['Sex'], Test_mod['Fare'], Test_mod['Parch'], Test_mod['SibSp'])), Test_mod['Age'])
Test = Test.apply(lambda row: row if np.isnan(row['Age']) == False else [Test['Pclass'][row.name], Test['Sex'][row.name],interp([Test['Pclass'][row.name], Test['Sex'][row.name], Test['Fare'][row.name], Test['Parch'][row.name], Test['SibSp'][row.name]])[0], Test['SibSp'][row.name], Test['Parch'][row.name], Test['Fare'][row.name], Test['Embarked'][row.name]], axis = 1)
X = X.apply(lambda row: row if np.isnan(row['Age']) == False else [X['Pclass'][row.name], X['Sex'][row.name],interp([X['Pclass'][row.name], X['Sex'][row.name], X['Fare'][row.name], X['Parch'][row.name], X['SibSp'][row.name]])[0], X['SibSp'][row.name], X['Parch'][row.name], X['Fare'][row.name], X['Embarked'][row.name]], axis = 1)
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
y_train.head()
print("X is",X,"Y is" ,y)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
model = Sequential()
model.add(Dense(units=32, activation='leaky_relu', input_dim=len(X_train.columns)))
model.add(Dense(units=64, activation='leaky_relu'))
model.add(Dense(units=32, activation='leaky_relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')
model.fit(X_train, y_train, epochs=1300, batch_size=32)
y_hat = model.predict(Test)
y_hat = [0 if val < 0.5 else 1 for val in y_hat]
print(len(y_hat))
Exit_data = {'PassengerId':[j for j in range(892, 1310)], 'Survived':y_hat} 
print(Exit_data['PassengerId'], len(Exit_data['Survived']))
Exit = pd.DataFrame(Exit_data)
Exit.to_csv('/home/ascundar/Downloads/titanic/Titanic_try3.csv', index = False)
y_hat = model.predict(X_test)
y_hat = [0 if val < 0.5 else 1 for val in y_hat]
print(accuracy_score(y_test, y_hat))