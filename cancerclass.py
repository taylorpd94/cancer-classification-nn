#Common Data Handlers
import pandas as pd
import numpy as np

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

#DATA
df = pd.read_csv("C:\\Users\\taylo\\tfnd\\DATA\\cancer_classification.csv")
df.info()
df.corr()['benign_0__mal_1'].drop('benign_0__mal_1').sort_values(ascending=True).plot(kind='bar')
sns.heatmap(df.corr(), annot=True, cmap='PiYG')

#CREATE VARIABLES
X = df.drop(['benign_0__mal_1'],axis=1).values
y = df['benign_0__mal_1'].values

#CREATE TRAIN/TEST SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#FEATURE SCALING
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#TENSORFLOW MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
model = Sequential()
early_stop = EarlyStopping(monitor='val_loss',mode='min',patience=20)
model.add(Dense(30, activation='relu'))
model.add(Dropout(.1))
model.add(Dense(15, activation='relu'))
model.add(Dropout(.1))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy')
model.fit(x=X_train,y=y_train,epochs=400, callbacks=[early_stop],validation_data=(X_test,y_test),batch_size=32)

#PREDICTIONS
y_pred = model.predict_classes(X_test)

#RESULTS
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
