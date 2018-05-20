# train with one dense LUT, test with another LUT
# there are no overlapping mua or musp values between the two LUTs

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
import numpy as np
import scipy.io as sio
import hdf5storage
from keras.callbacks import CSVLogger
import keras
from keras import regularizers
from keras.models import load_model


path = 'D:/Dropbox/Lab stuff/2018-01-10 - DNN inverse fitting/'

# lut_name = 'mat_two_LUTs_0_0p1_1p43_EqSp.mat'
lut_name = 'mat_two_LUTs_0_0p1_1p43_EqSp_extraTrainSet.mat'
# lut_name = 'DNN_train_LUT_EqSp_v01.mat'
# lut_name = 'DNN_train_LUT_linear_Rd_v01_generated_by_nearest.mat'


save_name = 'keras_SFDI_v02_twoOutputs.h5'

fname = path + lut_name
mat = hdf5storage.loadmat(fname)


train_set = mat['new_train_set']
# test_set = mat['test_set']

train_data = train_set[:, 0:2]
train_labels = train_set[:, 2:4]

# test_data = test_set[:, 0:2]
# test_labels = test_set[:, 2:4]



# num_neurons = 40
# beta = 0.0001
# drp_rate = 0.1
# # model
# model = Sequential()
# # model.add(Dense(200, input_dim=2))
# model.add(Dense(num_neurons, input_dim=2, activation='relu',  kernel_regularizer=regularizers.l2(beta), activity_regularizer=regularizers.l1(beta)))
# model.add(Activation('relu'))
# # model.add(Dropout(drp_rate))
# model.add(Dense(num_neurons,  kernel_regularizer=regularizers.l2(beta), activity_regularizer=regularizers.l1(beta)))
# model.add(Activation('relu'))
# # # model.add(Dropout(0.1))
# model.add(Dense(num_neurons,  kernel_regularizer=regularizers.l2(beta), activity_regularizer=regularizers.l1(beta)))
# model.add(Activation('relu'))
# # # model.add(Dropout(0.1))
# model.add(Dense(num_neurons,  kernel_regularizer=regularizers.l2(beta), activity_regularizer=regularizers.l1(beta)))
# model.add(Activation('relu'))
# # # model.add(Dropout(0.1))
# model.add(Dense(num_neurons,  kernel_regularizer=regularizers.l2(beta), activity_regularizer=regularizers.l1(beta)))
# model.add(Activation('relu'))
# # # model.add(Dropout(0.1))
# model.add(Dense(num_neurons,  kernel_regularizer=regularizers.l2(beta), activity_regularizer=regularizers.l1(beta)))
# model.add(Activation('relu'))
# # # model.add(Dropout(0.1))
# model.add(Dense(num_neurons,  kernel_regularizer=regularizers.l2(beta), activity_regularizer=regularizers.l1(beta)))
# model.add(Activation('relu'))
# # # model.add(Dropout(0.1))
# model.add(Dense(num_neurons,  kernel_regularizer=regularizers.l2(beta), activity_regularizer=regularizers.l1(beta)))
# model.add(Activation('relu'))
# # # # model.add(Dropout(0.1))
# model.add(Dense(2))


model = load_model(save_name)

# train model
optimizer = keras.optimizers.Adam(lr=0.0000001, beta_1=0.9, beta_2=0.999)
# optimizer = keras.optimizers.Adagrad(lr=0.0000000001, decay=0.0)
model.compile(loss='mean_absolute_percentage_error', optimizer=optimizer, metrics=['mape'])
# model.compile(loss='mae', optimizer=optimizer, metrics=['mae'])
# 3. fit the network
csv_logger = CSVLogger('log.csv')
model.fit(train_data, train_labels, epochs=10, batch_size=204800*1, callbacks=[csv_logger])
model.save(save_name)

#
# # evaluate model
# score_train = model.evaluate(train_data, train_labels, verbose=0)
# print('Train loss:', score_train[0])
# print('Train accuracy:', score_train[1])
#
# score_test = model.evaluate(test_data, test_labels, verbose=0)
# print('Test loss:', score_test[0])
# print('Test accuracy:', score_test[1])



#
# # 4. evaluate the network
# pred_X = model.predict(train_data)
# pred_X_test = model.predict(test_data)
#
# print(pred_X.shape)
#
# mape_train = np.mean(np.abs((train_labels - pred_X) / train_labels)) * 100
#
# print(mape_train.shape)
#
# mape_test = np.mean(np.abs((test_labels - pred_X_test) / test_labels)) * 100
# print('mean absolute percent error, train: ', mape_train)
# print('mean absolute percent error, test: ', mape_test)
#
# Result = {}
#
# Result['pred_X'] = pred_X
# Result['pred_X_test'] = pred_X_test
#
# Result['train_labels'] = train_labels
# Result['test_labels'] = test_labels
#
# sio.savemat(path+'keras_result_twoOutputs_temp', Result)
#
