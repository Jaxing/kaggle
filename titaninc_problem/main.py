from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas
import csv

training_data = pandas.read_csv('./train.csv')
test_data = pandas.read_csv('./test.csv')

training_data_input = (training_data.loc[:, ['Pclass', 'Sex']])\
                            .replace('female', 1)\
                            .replace('male', -1)
training_data_target = training_data.loc[:, 'Survived']


test_data_input = (test_data.loc[:, ['Pclass', 'Sex']])\
                        .replace('female', 1)\
                        .replace('male', -1)

print(training_data_input.shape, training_data_target.shape)

model = Sequential()
model.add(Dense(1, input_shape=(2,), activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(
    np.array(training_data_input),
    np.array(training_data_target),
    epochs=10,
    batch_size=10
)

online_prediction_model = Sequential()
online_prediction_model.add(Dense(1, input_shape=(2,), activation='sigmoid'))
online_prediction_model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


trained_weights = model.get_weights()
online_prediction_model.set_weights(trained_weights)
online_prediction_model.compile(optimizer='rmsprop', loss='mse')

result = [["PassengerId", "Survived"]]
for i in range(0, len(test_data_input)):
    testX = test_data_input.loc[i, :]
    passenger_id = test_data.loc[i, 'PassengerId']
    testX = testX.values.reshape(1, 2)
    yhat = online_prediction_model.predict(np.array(testX), batch_size=1)
    result.append([passenger_id, (1 if 0.5 < yhat[0][0] else 0)])
#
with open('submission.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for row in result:
        print(row)
        row = map(str, row)
        writer.writerow(row)
