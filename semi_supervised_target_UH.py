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
os.environ["CUDA_VISIBLE_DEVICES"]="4"

from keras import backend as K
cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = True

print('Loading data..')

data_all_s = sio.loadmat('/raid/ADDA/data/Simple_frames/simple_frames_UH_target.mat',mat_dtype=True)
x_all_s= data_all_s['frames']


x_all_s = (((x_all_s-x_all_s.min())/(x_all_s.max()-x_all_s.min()))-0.5)*2
# Scaling to [-1,1] 
y_all_s = data_all_s['label']

#### Removing class = 6 from data and leaving only 5 classes in total
a = np.where(y_all_s==6)
new_y_all_s = np.delete(y_all_s, a[0])
new_x_all_s = np.delete(x_all_s, a[0], axis=0)

x_all_s = new_x_all_s
y_all_s = new_y_all_s
##################################

X_all_s = np.reshape(x_all_s, x_all_s.shape + (1,)) # changing according to 
Y_all_s = np_utils.to_categorical(y_all_s - 1) # keras requirement, 5D and # 1 hot respectively

# nb_classes = 6 # For UH
nb_classes = 5 # For UH semi supervised with 5 classes

for run in range(1,10):

	samples = [16,24,32,40,48,56,64]
	# samples = [8]
	accuracy_each_run = [] 
	for num_samples in samples:

		numpy_file = np.load('/raid/ADDA/UH_3D_results/UH_3D_run_'+str(run)+'.npy')
		best_epoch = list(numpy_file).index(max(numpy_file))
	
		print('run :'+ str(run))
		print('num_samples :'+str(num_samples) )
		print('best_epoch:'+str(best_epoch))
		for i in range(1,nb_classes+1):
			class_ind = np.where(y_all_s==i)

			Xi_trn_s, Xi_val_test_s, Yi_trn_s, Yi_val_test_s = train_test_split(X_all_s[class_ind[0],:,:,:,:], Y_all_s[class_ind[0],:], train_size=0.6, random_state = run) # For UH
			
			print(Xi_trn_s.shape, Xi_val_test_s.shape, Yi_trn_s.shape, Yi_val_test_s.shape)

			Xi_trn_s, X_val, Yi_trn_s, Y_val = train_test_split(Xi_trn_s, Yi_trn_s, train_size=num_samples, random_state=run) # With UH

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
		print('Y_train_s shape', Y_train_s.shape)

		print('X_val_s shape:', X_val_s.shape)
		print('X_test_s shape:', X_test_s.shape)
	


		num_epochs = 20
		batch_size = num_samples
		learning_rate = 0.01
	

		model = Sequential()
	

		model.add(Conv3D(32, input_shape=(X_all_s.shape[1:]),kernel_size=(3, 3, 32), strides = (1,1,18), padding='same', name='l1', trainable = True))	
		model.add(BatchNormalization(name ='l2', trainable = True))
		model.add(Activation('relu'))
	
	
		model.add(Conv3D(32, kernel_size=(3, 3, 32), padding='same', strides = (1,1,18), name='l3', trainable = True))
		model.add(BatchNormalization(name ='l4', trainable = True))
		model.add(Activation('relu'))
	
		model.add(Conv3D(32,kernel_size=(3,3,32),padding = 'same', strides = (2,2,18), name ='l5', trainable = True))
		model.add(BatchNormalization(name='l6', trainable= True))
		model.add(Activation('relu'))
	
	
		model.add(Conv3D(64, kernel_size=(3, 3, 32), padding='same', strides = (1,1,18), name='l7', trainable = True))
		model.add(BatchNormalization(name = 'l8', trainable = True))
		model.add(Activation('relu'))
	
		model.add(Conv3D(64, kernel_size=(3, 3, 32), padding='same', strides = (1,1,18), name='l9', trainable = True))
		model.add(Activation('tanh'))
	
		model.add(Flatten())

		model.add(Dense(nb_classes, activation='softmax',name='classifier',trainable = True))
		model.summary()


		model.load_weights('/raid/ADDA/UH_new/UH/3D_target_weights_seed'+str(run)+'_epoch'+str(best_epoch),by_name=True)
		
		# source_model = load_model('/raid/ADDA/UH_new/UH/UH_simple_frames_source_model_seed'+str(run))
		
		# weight = source_model.layers[15].get_weights()[0]
		# biases = source_model.layers[15].get_weights()[1]
		# model.layers[15].set_weights((weight,biases))
		

		opt = Adam(lr=learning_rate)		
		
		model.compile(loss=categorical_crossentropy,optimizer=opt,metrics=['accuracy'])

		history = model.fit(X_train_s, Y_train_s, batch_size = batch_size, epochs=num_epochs, validation_data = (X_val_s, Y_val_s))	

		np.save('/raid/ADDA/logs/UH_SS-ADDA_20epochs_0p01/3D_SS-ADDA_BOT_seed_'+str(run)+'_smpls_'+str(num_samples)+'_train_accuracy', history.history['acc'])
		np.save('/raid/ADDA/logs/UH_SS-ADDA_20epochs_0p01/3D_SS-ADDA_BOT_seed_'+str(run)+'_smpls_'+str(num_samples)+'_val_accuracy', history.history['val_acc'])
		np.save('/raid/ADDA/logs/UH_SS-ADDA_20epochs_0p01/3D_SS-ADDA_BOT_seed_'+str(run)+'_smpls_'+str(num_samples)+'_train_loss', history.history['loss'])
		np.save('/raid/ADDA/logs/UH_SS-ADDA_20epochs_0p01/3D_SS-ADDA_BOT_seed_'+str(run)+'_smpls_'+str(num_samples)+'_val_loss', history.history['val_loss'])

		score, acc_test = model.evaluate(X_test_s, Y_test_s, batch_size= batch_size)
		print('Test score :', score)
		print('Test accuracy :', acc_test)
		accuracy_each_run.append(acc_test)
		# model.save_weights('/raid/ADDA/logs/UH_SS-ADDA_20epochs_0p01/3D_SS-ADDA_BOT_seed_'+str(run)+'_smpls_'+str(num_samples))
	np.save('/raid/ADDA/logs/UH_SS-ADDA_20epochs_0p01/3D_SS-ADDA_BOT_seed_'+str(run)+'_smpls_'+str(num_samples), accuracy_each_run)
	
	

