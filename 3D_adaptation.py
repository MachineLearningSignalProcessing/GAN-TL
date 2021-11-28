import sys
import numpy as np
import scipy.io as sio
import tensorflow as tf

import random
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, MaxPooling3D,Flatten,Conv3D, Reshape, AveragePooling3D, LeakyReLU,UpSampling3D,SpatialDropout3D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.losses import categorical_crossentropy
from keras. optimizers import Adam,SGD
from keras.layers.normalization import BatchNormalization
from keras.initializers import RandomNormal
import argparse
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#np.random.seed(1337)


from keras import backend as K
TF_CUDNN_WORKSPACE_LIMIT_IN_MB = 10


cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = True
K.set_session(K.tf.Session(config=cfg))

#Source DATA###################

print('Loading source data...')

data_all_s = sio.loadmat('/net/hico/data/users/nikhil/Global_SIP/Simple_frames/simple_frames_BOT5.mat',mat_dtype=True)
x_all_s = data_all_s['frames']

 # Only [-1,1] no otherthing done on data
x_all_s = (((x_all_s-x_all_s.min())/(x_all_s.max()-x_all_s.min()))-0.5)*2


y_all_s = data_all_s['label']


X_all_s = np.reshape(x_all_s, x_all_s.shape + (1,)) # changing according to 
Y_all_s = np_utils.to_categorical(y_all_s - 1) # keras requirement, 5D and 1 hot respectively


def train(BATCH_SIZE):
	nb_classes = 9
#acc_test = np.zeros(5)
	for run in range(0,1):
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
		print('Y_train_s shape:', Y_train_s.shape)
		print('Y_val_s shape:', Y_val_s.shape)
		print('Y_test_s.shape:', Y_test_s.shape)

		X_train_s, Y_train_s = shuffle(X_train_s,Y_train_s)

#print(X_train_s.shape)
#print(Y_train_s.shape)
##############################################



#Target Data #################################

		print('Loading target data...')

		data_all_t = sio.loadmat('/net/hico/data/users/nikhil/Global_SIP/Simple_frames/simple_frames_BOT7.mat',mat_dtype=True)
		x_all_t = data_all_t['frames']

		x_all_t = (((x_all_t-x_all_t.min())/(x_all_t.max()-x_all_t.min()))-0.5)*2

 # Only [-1,1] no otherthing done on data
		y_all_t = data_all_t['label']

		X_all_t = np.reshape(x_all_t, x_all_t.shape + (1,)) # changing according to 
		Y_all_t = np_utils.to_categorical(y_all_t - 1) # keras requirement, 5D and 	

		#X_train_t = X_all_t
		#Y_train_t = Y_all_t



		for i in range(1,nb_classes+1):
			class_ind = np.where(y_all_t==i)
			Xi_trn_t, Xi_val_test_t, Yi_trn_t, Yi_val_test_t = train_test_split(X_all_t[class_ind[0],:,:,:,:], Y_all_t[class_ind[0],:], train_size=64, random_state = run)
			Xi_val_t, Xi_tst_t, Yi_val_t, Yi_tst_t = train_test_split(Xi_val_test_t, Yi_val_test_t, train_size=0.5, random_state = run)
			if i==1:
				X_train_t, Y_train_t, X_val_t, Y_val_t, X_test_t, Y_test_t = Xi_trn_t, Yi_trn_t, Xi_val_t, Yi_val_t, Xi_tst_t, Yi_tst_t
			else:
				X_train_t = np.concatenate((X_train_t, Xi_trn_t), axis=0)
				Y_train_t = np.concatenate((Y_train_t, Yi_trn_t), axis=0)
				X_val_t = np.concatenate((X_val_t, Xi_val_t), axis =0)
				Y_val_t = np.concatenate((Y_val_t, Yi_val_t), axis=0)
				X_test_t = np.concatenate((X_test_t, Xi_tst_t), axis=0)
				Y_test_t = np.concatenate((Y_test_t, Yi_tst_t), axis=0)

		print('X_train_t shape:', X_train_t.shape)
		print('X_val_t shape:', X_val_t.shape)
		print('X_test_t shape:', X_test_t.shape)

		X_train_t, Y_train_t = shuffle(X_train_t,Y_train_t)

#print(X_train_t.shape)
#print(Y_train_t.shape)

	
#####################################################################



####Source Model####################################################

		def source_model():	
			model = Sequential()
	
			model.add(Conv3D(32, input_shape=(X_all_s.shape[1:]),kernel_size=(3, 3, 32), strides = (1,1,18), padding='same', name='l1'))	
			model.add(BatchNormalization(name ='l2'))
			model.add(Activation('relu'))
	
	
			model.add(Conv3D(32, kernel_size=(3, 3, 32), padding='same', strides = (1,1,18), name='l3'))
			model.add(BatchNormalization(name ='l4'))
			model.add(Activation('relu'))
	
			model.add(Conv3D(32,kernel_size=(3,3,32),padding = 'same', strides = (2,2,18), name ='l5'))
			model.add(BatchNormalization(name='l6'))
			model.add(Activation('relu'))
	
	
			model.add(Conv3D(64, kernel_size=(3, 3, 32), padding='same', strides = (1,1,18), name='l7'))
			model.add(BatchNormalization(name = 'l8'))
			model.add(Activation('relu'))
	
			model.add(Conv3D(64, kernel_size=(3, 3, 32), padding='same', strides = (1,1,18), name='l9'))
			model.add(Activation('tanh'))

			return model


#######################################################################

### Target Model######################################################

		def target_model():	
			model = Sequential()
	
			model.add(Conv3D(32, input_shape=(X_all_s.shape[1:]),kernel_size=(3, 3, 32), strides = (1,1,18), padding='same', name='l1'))	
			model.add(BatchNormalization(name ='l2'))
			model.add(Activation('relu'))
	
	
			model.add(Conv3D(32, kernel_size=(3, 3, 32), padding='same', strides = (1,1,18), name='l3'))
			model.add(BatchNormalization(name ='l4'))
			model.add(Activation('relu'))
	
			model.add(Conv3D(32,kernel_size=(3,3,32),padding = 'same', strides = (2,2,18), name ='l5'))
			model.add(BatchNormalization(name='l6'))
			model.add(Activation('relu'))
	
	
			model.add(Conv3D(64, kernel_size=(3, 3, 32), padding='same', strides = (1,1,18), name='l7'))
			model.add(BatchNormalization(name = 'l8'))
			model.add(Activation('relu'))
	
			model.add(Conv3D(64, kernel_size=(3, 3, 32), padding='same', strides = (1,1,18), name='l9'))
			model.add(Activation('tanh'))


			return model

################################################################################

### Discriminator model ###############################################

		def discriminator_model():	
			model = Sequential()

			model.add(Conv3D(32, input_shape=(4,4,1,64),kernel_size=(3, 3, 32), padding='same',name = 'l1'))
			model.add(LeakyReLU(alpha=0.2))
		 
			model.add(Conv3D(32, kernel_size=(3, 3, 32), padding = 'same', name = 'l2'))
			model.add(BatchNormalization(name ='l3'))
			model.add(LeakyReLU(alpha=0.2))
	
			model.add(Conv3D(64, kernel_size=(3, 3, 32), padding = 'same', name = 'l4'))
			model.add(BatchNormalization(name = 'l5'))
			model.add(LeakyReLU(alpha=0.2))
		
			model.add(Conv3D(64, kernel_size=(3, 3, 32), padding = 'same', name = 'l6'))
			model.add(BatchNormalization(name = 'l7'))
			model.add(LeakyReLU(alpha=0.2))
	
			model.add(Conv3D(64, kernel_size=(3, 3, 32), padding='same', name = 'l8'))
			model.add(BatchNormalization(name = 'l9'))
			model.add(LeakyReLU(alpha=0.2))
	
			model.add(Flatten())
	
			model.add(Dense(1))
			model.add(Activation('sigmoid'))
	


			return model	

#########################################################################
### Target containing discriminator model###############################

		def target_containing_discriminator(t,d):
			model = Sequential()
			model.add(t)
			model.add(d)
			d.trainable = False
			return model


#####################################################
		d = discriminator_model()
		s = source_model()
		t = target_model()
		d_on_t = target_containing_discriminator(t,d)

 



		lr = 0.00001
 	
		s.load_weights('/net/hico/data/users/nikhil/Global_SIP/3D/simple_frames_source_weights_seed'+str(run), by_name = True)
		t.load_weights('/net/hico/data/users/nikhil/Global_SIP/3D/simple_frames_source_weights_seed'+str(run), by_name = True)
	
	
		d_optim = Adam(lr=0.00001, beta_1 = 0.5)
	
		d.trainable = True
	
		d.compile(loss='binary_crossentropy',optimizer = d_optim)
		s.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.00001))

		d_on_t_optim = Adam(lr=0.00001, beta_1=0.5, beta_2=0.999, epsilon = 1e-08)
		d_on_t.compile(loss = 'binary_crossentropy', optimizer = d_on_t_optim)

		d_on_t_loss_plot = np.zeros(100)
		d_loss_plot = np.zeros(100)
		for epoch in range(100):
		

			print("\n")
			print('Epoch is', epoch)
			print('Number of batches', int(X_train_t.shape[0]/BATCH_SIZE))

			source_generated_images = s.predict(X_train_s,verbose=0)

			
		
			for index in range(int(X_train_t.shape[0]/BATCH_SIZE)):
				target_generated_images = t.predict(X_train_t, verbose =0)
			
				source_image_batch = source_generated_images[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
				target_image_batch = target_generated_images[index*BATCH_SIZE:(index+1)*BATCH_SIZE]

				noisy_source_image_batch = source_image_batch + np.random.normal( 0, 0.1,[BATCH_SIZE,4,4,1,64])
				noisy_target_image_batch = target_image_batch + np.random.normal( 0, 0.1,[BATCH_SIZE,4,4,1,64])

			
				d_loss_real = d.train_on_batch(noisy_source_image_batch,np.asarray([0.95]*BATCH_SIZE))
			
				d_loss_fake = d.train_on_batch(noisy_target_image_batch,np.asarray([0]*BATCH_SIZE))
				d_loss = np.add(d_loss_real,d_loss_fake)
			

				print('batch %d d_loss : %f' %(index, d_loss))
				d.trainable = False
				X_train_batch = X_train_t[index*BATCH_SIZE:(index+1)*BATCH_SIZE]

			
				t_loss = d_on_t.train_on_batch(X_train_batch, np.asarray([0.95]*BATCH_SIZE))
			
				d.trainable = True


				print('batch %d t_loss : %f' % (index, t_loss))
				print('\n')
		
			

			t.save_weights('/net/hico/data/users/nikhil/Global_SIP/3D/3D_BOT_weights/3D_target_weights_seed'+str(run)+'_epoch'+str(epoch),True)
			d_loss_plot[epoch] = d_loss
			d_on_t_loss_plot[epoch] = t_loss

		fig, ax = plt.subplots()
		ax.plot(d_loss_plot, 'r', label = 'd_loss')
		ax.plot(d_on_t_loss_plot, 'b', label = 't_loss')
		ax.legend()
		#plt.show()
		plt.savefig('/net/hico/data/users/nikhil/Global_SIP/final_BOT_3D_losses.png')
def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode',type=str)
	parser.add_argument('--batch_size', type= int, default =128)
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = get_args()
	if args.mode == 'train':
		train(BATCH_SIZE = args.batch_size)


	

