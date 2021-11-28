import sys
import numpy as np
import scipy.io as sio

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, MaxPooling3D,Flatten,Conv3D,AveragePooling3D, LeakyReLU
from sklearn.model_selection import train_test_split
from keras.losses import categorical_crossentropy
from keras. optimizers import Adam, SGD
from keras.layers.normalization import BatchNormalization

import os 
os.environ["CUDA_VISIBLE_DEVICES"]="3"

#

from keras import backend as K
cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = True
K.set_session(K.tf.Session(config=cfg))

print('Loading data...')

data_all = sio.loadmat('/net/hico/data/users/nikhil/Global_SIP/Simple_frames/simple_frames_BOT5.mat',mat_dtype=True)
x_all_s = data_all['frames']

x_all_s = (((x_all_s-x_all_s.min())/(x_all_s.max()-x_all_s.min()))-0.5)*2

y_all_s = data_all['label']

X_all_s = np.reshape(x_all_s, x_all_s.shape + (1,)) # changing according to 
Y_all_s = np_utils.to_categorical(y_all_s- 1) # keras requirement, 5D and # 1 hot respectively

nb_classes = 9
acc_test = np.zeros(10)

run = 0
for run in range(0,1):
	np.random.seed(run)
	print('run :'+ str(run))
	for i in range(1,nb_classes+1):
		class_ind = np.where(y_all_s==i)
		Xi_trn_s, Xi_val_test_s, Yi_trn_s, Yi_val_test_s = train_test_split(X_all_s[class_ind[0],:,:,:,:], Y_all_s[class_ind[0],:], train_size =64 , random_state = run)
		Xi_val_s, Xi_tst_s, Yi_val_s, Yi_tst_s = train_test_split(Xi_val_test_s, Yi_val_test_s, train_size=0.5, random_state = run)
    		
		if i==1:
			X_train_s, Y_train_s, X_val_s, Y_val_s, X_test_s, Y_test_s = Xi_trn_s, Yi_trn_s, Xi_val_s, Yi_val_s, Xi_tst_s, Yi_tst_s
		else:
			X_train_s = np.concatenate((X_train_s, Xi_trn_s), axis=0)
			Y_train_s = np.concatenate((Y_train_s, Yi_trn_s), axis=0)
			X_val_s = np.concatenate((X_val_s, Xi_val_s), axis =0)
			Y_val_s = np.concatenate((Y_val_s, Yi_val_s), axis=0)
			X_test_s = np.concatenate((X_test_s, Xi_tst_s), axis=0)
			Y_test_s = np.concatenate((Y_test_s, Yi_tst_s), axis=0)

	print('X_train_s shape:', X_train_s.shape)	
	print('X_val_s shape:', X_val_s.shape)
	print('X_test_s shape:', X_test_s.shape)
	
#X_train_s, Y_train_s = shuffle(X_train_s,Y_train_s)

	num_epochs = 100
	batch_size = 64
		
	model = Sequential()

	model.add(Conv3D(32, input_shape=(X_all_s.shape[1:]),kernel_size=(3, 3, 32), strides = (1,1,18), padding='same', name='l1',trainable=True))	
	model.add(BatchNormalization(name ='l2',trainable = True))
	model.add(Activation('relu'))
	
	
	model.add(Conv3D(32, kernel_size=(3, 3, 32), padding='same', strides = (1,1,18), name='l3',trainable = True))
	model.add(BatchNormalization(name ='l4',trainable = True))
	model.add(Activation('relu'))
	
	model.add(Conv3D(32,kernel_size=(3,3,32),padding = 'same', strides = (2,2,18), name ='l5',trainable = True))
	model.add(BatchNormalization(name='l6',trainable =True))
	model.add(Activation('relu'))
	
	
	model.add(Conv3D(64, kernel_size=(3, 3, 32), padding='same', strides = (1,1,18), name='l7',trainable= True))
	model.add(BatchNormalization(name = 'l8',trainable =True))
	model.add(Activation('relu'))

	model.add(Conv3D(64, kernel_size=(3, 3, 32), padding='same', strides = (1,1,18), name='l9',trainable = True))
	model.add(Activation('tanh'))

	model.add(Flatten())
	model.add(Dense(nb_classes, activation='softmax',name = 'l10'))
	print(model.summary())
	opt = Adam(lr=0.00001)

	model.compile(loss=categorical_crossentropy,optimizer=opt,metrics=['accuracy'])
		
	history = model.fit(X_train_s, Y_train_s,batch_size=batch_size, epochs=num_epochs, validation_data = (X_val_s, Y_val_s))
	
	score, acc_test[run] = model.evaluate(X_test_s,Y_test_s, batch_size=batch_size)
	
	print('Test score:', score)
	print('Test accuracy:', acc_test[run])
	model.save_weights('/net/hico/data/users/nikhil/Global_SIP/3D/simple_frames_source_weights_seed'+str(run))
	model.save('/net/hico/data/users/nikhil/Global_SIP/3D/simple_frames_source_model_seed'+str(run))

	print('Test Accuracy stride 18 :' + str(acc_test))
	print('Mean: '+ str(np.mean(acc_test)))
	print('std: '+ str(np.std(acc_test)))	

	

