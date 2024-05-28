import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings('ignore')

path = r'KerasClassification\cancer_classification.csv'
df = pd.read_csv(path)
print(df.head())

# EDA
# print(df.info())
# sns.heatmap(df.corr(), annot=True, cmap='Spectral')
# print(df.corr())
# df.corr()['benign_0__mal_1'].iloc[:-1].sort_values().plot(kind='bar')
# plt.show()

# modeling
x = df.drop(["benign_0__mal_1"], axis=1)
y = df['benign_0__mal_1']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=101)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(type(x_train))
print(x_train.shape)
print(x_train)

model = Sequential()
model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # because of binary classification

model.compile(loss='binary_crossentropy', optimizer='adam')
# hist = model.fit(x_train, y_train, epochs=600,verbose=1, validation_data=(x_test, y_test)) # epoch is high overfitting

# loss_overfit = pd.DataFrame(hist.history)
# loss_overfit.to_csv('KerasClassificationOverfittingLoss.csv', index=False)
# loss_overfit.plot() # as we see val_loss increase training loss decrease so model overfitting occur
# loss_overfit = pd.read_csv('Basics of Keras\KerasClassificationOverfittingLoss.csv')
# print(loss_overfit.head())
# plt.show()


model = Sequential()
model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')

# to stop the overfitting we choose the early stoppig mechanism
# help
# print(help(EarlyStopping))
# early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25) # patience=25 means we wait 25 epochs after we detect stopping point beacause of noise 
# monitor parameter for the what we want monitor and mode parameter 'min' means we want to minimize mmonitor parameter
# hist = model.fit(x_train, y_train, epochs=600, verbose=1, validation_data=(x_test, y_test), callbacks=[early_stop])
# stop after 97 epochs

# loss_early_stop = pd.DataFrame(hist.history)
# loss_early_stop.to_csv('KerasClassificationEarlyStoppingLoss.csv', index=False)
# loss_early_stop.plot()
# plt.show()

model = Sequential()
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.5)) # 0.5 means 50% neuron off in each batch of traininng here whole epoch because we didnt specify the batch 
model.add(Dense(15, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
# hist = model.fit(x_train, y_train, epochs=600, verbose=1, validation_data=(x_test, y_test), callbacks=[early_stop])

# loss_Dropout_early_stop = pd.DataFrame(hist.history)
# loss_Dropout_early_stop.to_csv('KerasClassificationDropoutEarlyStoppingLoss.csv', index=False)
# loss_Dropout_early_stop.plot()
# plt.show()

# all time best
# model.save('KerasClassification.h5')
best_model = load_model('KerasClassification\KerasClassification.h5')

predictions = best_model.predict_classes(x_test).reshape(143,) 
dict = {'Actual':y_test, 'Predict':predictions}
compare = pd.DataFrame(dict)
print(compare.sample(10))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))



