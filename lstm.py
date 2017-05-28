import numpy as np
import random
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Masking
from keras.layers.recurrent import LSTM

# dataのshape (3, x, 4*4*512)
# (None, None, 4*4*512)

random.seed(0)

max = 79

# load data
X_liz1 = np.load('./dataset/results/Liz/features_liz_1.npy')
X_liz1 = X_liz1.reshape(1, -1, 8192)

X_liz2 = np.load('./dataset/results/Liz/features_liz_2.npy')
X_tmp = np.zeros((max - X_liz2.shape[0], 8192))
X_liz2 = np.vstack([X_liz2, X_tmp]).reshape(1, -1, 8192)

X_liz3 = np.load('./dataset/results/Liz/features_liz_3.npy')
X_tmp = np.zeros((max - X_liz3.shape[0], 8192))
X_liz3 = np.vstack([X_liz3, X_tmp]).reshape(1, -1, 8192)

X_naomi1 = np.load('./dataset/results/Naomi/features_naomi_1.npy')
X_tmp = np.zeros((max - X_naomi1.shape[0], 8192))
X_naomi1 = np.vstack([X_naomi1, X_tmp]).reshape(1, -1, 8192)

X_naomi2 = np.load('./dataset/results/Naomi/features_naomi_2.npy')
X_tmp = np.zeros((max - X_naomi2.shape[0], 8192))
X_naomi2 = np.vstack([X_naomi2, X_tmp]).reshape(1, -1, 8192)

X_naomi3 = np.load('./dataset/results/Naomi/features_naomi_3.npy')
X_tmp = np.zeros((max - X_naomi3.shape[0], 8192))
X_naomi3 = np.vstack([X_naomi3, X_tmp]).reshape(1, -1, 8192)

X_tyler1 = np.load('./dataset/results/Tyler/features_tyler_1.npy')
X_tmp = np.zeros((max - X_tyler1.shape[0], 8192))
X_tyler1 = np.vstack([X_tyler1, X_tmp]).reshape(1, -1, 8192)

X_tyler2 = np.load('./dataset/results/Tyler/features_tyler_2.npy')
X_tmp = np.zeros((max - X_tyler2.shape[0], 8192))
X_tyler2 = np.vstack([X_tyler2, X_tmp]).reshape(1, -1, 8192)

X_tyler3 = np.load('./dataset/results/Tyler/features_tyler_3.npy')
X_tmp = np.zeros((max - X_tyler3.shape[0], 8192))
X_tyler3 = np.vstack([X_tyler3, X_tmp]).reshape(1, -1, 8192)

X_train = []
X_test = []


#X_test.append(X_liz1)
#X_test.append(X_naomi2)
#X_test.append(X_tyler3)
#X_test = np.array(X_test)
X_test = np.vstack([X_liz1, X_naomi2, X_tyler3])

y_test = np.array([1, 2, 3], dtype=np.float32)

#X_train.append(X_liz2)
#X_train.append(X_liz3)
#X_train.append(X_naomi1)
#X_train.append(X_naomi3)
#X_train.append(X_tyler1)
#X_train.append(X_tyler2)

X_train = np.vstack([X_liz2, X_liz3, X_naomi1, X_naomi3, X_tyler1, X_tyler2])

y_train = np.array([2, 3, 1, 3, 1, 2], dtype=np.float32)


# Keras
in_out_neurons = 1
hidden_neurons = 512

model = Sequential()
model.add(Masking(mask_value=0., input_shape=(max, 8192)))
model.add(LSTM(hidden_neurons, batch_input_shape=(None, None, 8192), return_sequences=False))
model.add(Dense(in_out_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")
model.fit(X_train, y_train, batch_size=600, nb_epoch=100, verbose=1, validation_split=0.05)

predicted = model.predict(X_test)
print(predicted)
