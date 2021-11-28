import sys
import numpy as np
import scipy.io as sio

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, MaxPooling3D,Flatten,Conv3D, AveragePooling3D,LeakyReLU,SpatialDropout3D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.losses import categorical_crossentropy
from keras. optimizers import Adam, SGD
from keras.models import Model, load_model
from keras.layers.normalization import BatchNormalization
import matplotlib
from sklearn.metrics import confusion_matrix
#matplotlib.use('agg')
import matplotlib.pyplot as plt

#np.random.seed(1337)
np.random.seed(10)
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from keras import backend as K
cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = True
K.set_session(K.tf.Session(config=cfg))

print('Loading data..')

data_all_s = sio.loadmat('/net/hico/data/users/nikhil/Global_SIP/Simple_frames/simple_frames_BOT7.mat',mat_dtype=True)
x_all_s= data_all_s['frames']


x_all_s = (((x_all_s-x_all_s.min())/(x_all_s.max()-x_all_s.min()))-0.5)*2


 # Only [-1,1] no otherthing done on data

y_all_s = data_all_s['label']

X_all_s = np.reshape(x_all_s, x_all_s.shape + (1,)) # changing according to 
Y_all_s = np_utils.to_categorical(y_all_s - 1) # keras requirement, 5D and # 1 hot respectively

nb_classes = 9
acc_test = np.zeros(10)


for run in range(0,5):
	
	print('run :'+ str(run))
	for i in range(1,nb_classes+1):
		class_ind = np.where(y_all_s==i)
		Xi_trn_s, Xi_val_test_s, Yi_trn_s, Yi_val_test_s = train_test_split(X_all_s[class_ind[0],:,:,:,:], Y_all_s[class_ind[0],:], train_size=64, random_state = run)
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
	


	num_epochs = 100
	batch_size = 64
	learning_rate = 0.00001
	acc_plot = np.zeros(num_epochs)

	model = Sequential()
	

	model.add(Conv3D(32, input_shape=(X_all_s.shape[1:]),kernel_size=(3, 3, 32), strides = (1,1,18), padding='same', name='l1', trainable = False))	
	model.add(BatchNormalization(name ='l2', trainable = False))
	model.add(Activation('relu'))
	
	
	model.add(Conv3D(32, kernel_size=(3, 3, 32), padding='same', strides = (1,1,18), name='l3', trainable = False))
	model.add(BatchNormalization(name ='l4', trainable = False))
	model.add(Activation('relu'))
	
	model.add(Conv3D(32,kernel_size=(3,3,32),padding = 'same', strides = (2,2,18), name ='l5', trainable = False))
	model.add(BatchNormalization(name='l6', trainable= False))
	model.add(Activation('relu'))
	
	
	model.add(Conv3D(64, kernel_size=(3, 3, 32), padding='same', strides = (1,1,18), name='l7', trainable = False))
	model.add(BatchNormalization(name = 'l8', trainable = False))
	model.add(Activation('relu'))
	
	model.add(Conv3D(64, kernel_size=(3, 3, 32), padding='same', strides = (1,1,18), name='l9', trainable = False))
	model.add(Activation('tanh'))
	
	model.add(Flatten())

	model.add(Dense(nb_classes, activation='softmax',trainable = False))
	model.summary()
	

	for i in range (0,num_epochs):

		model.load_weights('/net/hico/data/users/nikhil/Global_SIP/3D/3D_BOT_weights/3D_target_weights_seed'+str(run)+'_epoch'+str(i),by_name=True)
	
		source_model = load_model('/net/hico/data/users/nikhil/Global_SIP/3D/simple_frames_source_model_seed'+str(run))
		
		weight = source_model.layers[15].get_weights()[0]
		biases = source_model.layers[15].get_weights()[1]
		model.layers[15].set_weights((weight,biases))
		

		
		opt = Adam(lr=learning_rate, beta_1=0.5, beta_2=0.999, epsilon = 1e-08)
		
		model.compile(loss=categorical_crossentropy,optimizer=opt,metrics=['accuracy'])
	
		score, acc_test[run] = model.evaluate(X_test_s,Y_test_s, batch_size=batch_size)
		predictions = model.predict(X_test_s)
		cm = confusion_matrix(np.argmax(Y_test_s,axis=1),np.argmax(predictions,axis=1))
		print('confusion matrix : ', cm)
		print('Test score:', score)
		print('Test accuracy:', acc_test[run])
		acc_plot[i] = acc_test[run]
	

	#print('Accuracies for '+str(run), acc_plot)
	#plt.plot(acc_plot)
	#plt.title('Accuracy - epoch')
	#plt.ylabel('Accuracy')
	#plt.xlabel('Epochs')
	#plt.show()
	#plt.savefig('/net/hico/data/users/nikhil/Global_SIP/final_BOT_3D_acc.png')

print('Test Accuracy:' + str(acc_test))
print('Mean: '+ str(np.mean(acc_test)))
print('std: '+ str(np.std(acc_test)))
