
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler

model = Sequential([

    Dense(30, input_shape=(13,)),
    Activation("relu"),
    BatchNormalization(),
    Dropout(.4),

    Dense(25),
    Activation("relu"),
    BatchNormalization(),
    Dropout(.4),

    Dense(20),
    Activation("relu"),
    BatchNormalization(),
    Dropout(.4),

    Dense(15),
    Activation("relu"),
    BatchNormalization(),
    Dropout(.4),

    Dense(8),
    Activation("relu"),
    BatchNormalization(),
    Dropout(.4),

    Dense(5),
    Activation("relu"),
    BatchNormalization(),
    Dropout(.4),

    Dense(1)
])

model.compile(optimizer=Adam(lr=.001), loss="mse")

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

in_scaler = StandardScaler()
out_scaler = StandardScaler()
in_scaler.fit(x_train)
x_train = in_scaler.transform(x_train)
x_test = in_scaler.transform(x_test)
y_train = y_train.reshape((x_train.shape[0], 1))
y_test = y_test.reshape((x_test.shape[0], 1))
out_scaler.fit(y_train)
y_train = out_scaler.transform(y_train)
y_test = out_scaler.transform(y_test)

res = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1000, batch_size=64)

print(len(res.history["loss"]))

