import tensorflow as tf 
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping,LearningRateScheduler,CSVLogger,TensorBoard,ModelCheckpoint

import pickle
import os
import numpy as np
from data_helper import helper
from model_utils import build_model

class make_model:

	def __init__(self,checkpoint_dir,checkpoint_addr,log_csv_addr,tensorboard_logs,
		train_keys_path,path_to_data,val_frac,final_model_path,mode,threshold_path,restore):


		print("\nTRAINING FOR {} MODE -----\n".format(mode))

		timesteps=5
		data_loader=helper(path_to_data,val_frac=val_frac,timesteps=timesteps,shift=2)
		build_keys={'input_shape':(timesteps,data_loader.X_train.shape[2]),'mode':mode,'timesteps':timesteps}
		self.train_generator,self.val_generator=data_loader.generator_train_val(train_keys_path=train_keys_path,restore=restore,mode=build_keys['mode'])
		self.model=build_model(build_keys)
		self.num_batches_train=data_loader.num_batches_train
		self.num_batches_val=data_loader.num_batches_val

		self.checkpoint_dir=checkpoint_dir
		self.checkpoint_addr=checkpoint_addr
		self.log_csv_addr=log_csv_addr
		self.tensorboard_logs=tensorboard_logs
		self.final_model_path=final_model_path

		self.neg_data_yielder_val=data_loader.single_yielder(data_loader.X_val_y0,batch_size=32)
		self.pos_data_yielder_val=data_loader.single_yielder(data_loader.X_val_y1,batch_size=32)

		print(data_loader.X_val_y1.shape)

		self.neg_data_yielder_train=data_loader.single_yielder(data_loader.X_train_y0,batch_size=32)
		self.pos_data_yielder_train=data_loader.single_yielder(data_loader.X_train_y1,batch_size=32)
		self.threshold_path=threshold_path

	def lr_schedule(self,curr_epoch,curr_lr):
		if (curr_epoch+1)%50==0:	
			return curr_lr/2
		else:
			return curr_lr

	def get_callbacks(self):
		if not os.path.exists(self.checkpoint_dir):
			os.mkdir(self.checkpoint_dir)

		checkpoint = ModelCheckpoint(self.checkpoint_addr, monitor='val_loss', verbose=1, 
			save_best_only=True, mode='min')
		earlystopping=EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='min', baseline=None, 
			restore_best_weights=True)
		lrschedule=LearningRateScheduler(self.lr_schedule, verbose=1)
		csvlog=CSVLogger(self.log_csv_addr, append=False)
		tensorboard=TensorBoard(log_dir=self.tensorboard_logs, write_graph=True,update_freq=100)

		self.callbacks_list = [checkpoint,lrschedule,csvlog,earlystopping,tensorboard]

	def summarize(self):
		print(self.model.summary())
		plot_model(self.model,to_file='model.png')

	def compile_model(self):
		optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
		loss=MeanSquaredError()
		self.model.compile(optimizer=optimizer,loss=loss)

	def train_model(self):
		self.get_callbacks()
		self.compile_model()
		self.summarize()

		if os.path.exists(checkpoint_addr):
			self.model.load_weights(checkpoint_addr,by_name=True)
			print('LOADED MODEL WEIGHTS')
		else:
			print('TRAINING FROM SCRATCH')

		self.history=self.model.fit(self.train_generator,epochs=300,steps_per_epoch=self.num_batches_train,validation_data=self.val_generator,
			validation_steps=self.num_batches_val,verbose=1,callbacks=self.callbacks_list)
		
		self.export_model()

	def train_on_batch(self):
		self.get_callbacks()
		self.compile_model()
		self.summarize()

		if os.path.exists(checkpoint_addr):
			self.model.load_weights(checkpoint_addr,by_name=True)
			print('LOADED MODEL WEIGHTS')
		else:
			print('TRAINING FROM SCRATCH')

		last=1
		for x in range(1000):
			X,_,_=next(self.train_generator)
			history=self.model.train_on_batch(X,X)
			if history-last>2:
				print(x)
				print("+"*50)
			last=history
			print(history)

		self.export_model()


	def export_model(self):
		self.model.save(self.final_model_path)

	def reconstruction_loss(self):
		rse_loss_y1_train=self.model.evaluate(self.pos_data_yielder_train)
		rse_loss_y0_train=self.model.evaluate(self.neg_data_yielder_train)

		rse_loss_y1_val=self.model.evaluate(self.pos_data_yielder_val)
		rse_loss_y0_val=self.model.evaluate(self.neg_data_yielder_val)


		print("RECONSTRUCTION LOSSES :: ")
		print('-'*50)
		print("Train rse_loss_y0 ",rse_loss_y0_train)
		print("Train rse_loss_y1 ",rse_loss_y1_train)
		print("Val rse_loss_y0 ",rse_loss_y0_val)
		print("Val rse_loss_y1 ",rse_loss_y1_val)

		threshold=rse_loss_y1_train
		print("Choose the threshold as between train losses of y1 and y0 = {}".format(threshold))

		with open(self.threshold_path,'wb+') as f:
			pickle.dump(threshold,f)

if __name__=='__main__':

	modes=['ae_rse','rse']

	for mode in modes:

		output_data_path=os.path.join('..','output_data_{}'.format(mode))
		if not os.path.exists(output_data_path):
			os.mkdir(output_data_path)

		path_to_data=os.path.join('..','data','rare_events.csv')
		train_keys_path=os.path.join(output_data_path,'train_keys.pkl')
		checkpoint_dir=os.path.join(output_data_path,'checkpoints')
		checkpoint_addr=os.path.join(checkpoint_dir,'weightsbest.hdf5')
		log_csv_addr=os.path.join(output_data_path,'training_log.csv')
		tensorboard_logs=os.path.join(output_data_path,'tboard_logs')
		final_model_path=os.path.join(output_data_path,'h5_file','model.h5')
		threshold_path=os.path.join(output_data_path,'threshold.pkl')
		if not os.path.exists(os.path.join(output_data_path,'h5_file')):
			os.mkdir(os.path.join(output_data_path,'h5_file'))

		val_frac=0.1

		restore=False
		if os.path.exists(checkpoint_addr):
			restore=True

		mymodel=make_model(checkpoint_dir=checkpoint_dir,checkpoint_addr=checkpoint_addr,log_csv_addr=log_csv_addr,
			tensorboard_logs=tensorboard_logs,mode=mode,threshold_path=threshold_path,train_keys_path=train_keys_path,path_to_data=path_to_data,val_frac=val_frac,final_model_path=final_model_path,restore=restore)
		mymodel.train_model()
		mymodel.reconstruction_loss()


