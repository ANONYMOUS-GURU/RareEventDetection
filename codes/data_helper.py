import pandas as pd 
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
def make_temporal_features(X,y,timesteps):
	temporal_X=[]
	temporal_y=[]
	for x in range(X.shape[0]-timesteps+1):
		temporal_X.append(X[x:x+timesteps])
		temporal_y.append(y[x+timesteps-1])
	return np.asarray(temporal_X),np.asarray(temporal_y)

class helper:
	def __init__(self,PATH,timesteps,val_frac=0.1,shift=2):
		
		data=pd.read_csv(PATH)

		data['time']=pd.to_datetime(data['time'])
		data.columns=list(data.columns[:-1])+['target']
		data=data.sort_values(by='time')

		data['shifted_target']=0
		for x in range(data.shape[0]):
			if data['target'].iloc[x]==1:
				data['shifted_target'].iloc[x-shift:x]=1

		data.drop(data[data['target']==1].index,inplace=True)
		data.drop(['time','target'],axis=1,inplace=True)

		X=data[data.columns[:-1]].values
		y=data['shifted_target'].values

		self.train_keys={'standardscaler':None,'timesteps':timesteps}

		X,y=make_temporal_features(X,y,self.train_keys['timesteps'])

		self.X_train,self.X_val,self.y_train,self.y_val=train_test_split(X,y,test_size=val_frac,random_state=0)

		self.batch_size_train=128
		self.batch_size_val=2048

		self.X_train_y0=self.X_train[self.y_train==0]
		self.X_train_y1=self.X_train[self.y_train==1]

		print("shape y0 ",self.X_train_y0.shape)

		self.X_val_y0=self.X_val[self.y_val==0]
		self.X_val_y1=self.X_val[self.y_val==1]

		self.num_batches_train=(int)(np.ceil(self.X_train_y0.shape[0]/self.batch_size_train))
		self.num_batches_val=(int)(np.ceil(self.X_val_y0.shape[0]/self.batch_size_val))


	def yielder_autoencoder(self,data,batch_size):
		i=0
		while True:
			start=i*batch_size
			end=min((i+1)*batch_size,data.shape[0])
			# print("start = {} ,, end = {}   shape = {} ".format(start,end,data.shape))
			return_batch=data[start:end]
			i+=1
			return_batch_=[]
			for x in range(return_batch.shape[0]):
				return_batch_.append(self.train_keys['standardscaler'].transform(return_batch[x]))

			return_batch_=np.asarray(return_batch_)
			yield return_batch_,return_batch_,[None]

			if end==data.shape[0]:
				i=0

	def single_yielder(self,data,batch_size):
		i=0
		end=0
		while end<data.shape[0]:
			start=i*batch_size
			end=min((i+1)*batch_size,data.shape[0])
			return_batch=data[start:end]
			i+=1
			return_batch_=[]
			for x in range(return_batch.shape[0]):
				return_batch_.append(self.train_keys['standardscaler'].transform(return_batch[x]))

			return_batch_=np.asarray(return_batch_)
			yield return_batch_,return_batch_,[None]


	def generator_train_val(self,train_keys_path,restore=False,mode='ae_rse'):

		if not restore:
			self.get_train_keys(self.X_train_y0)
			self.save_train_keys(train_keys_path)
		else:
			self.restore_train_keys(train_keys_path)

		return self.yielder_autoencoder(data=self.X_train_y0,batch_size=self.batch_size_train),self.yielder_autoencoder(data=self.X_val_y0,batch_size=self.batch_size_val)

	# # FUNCTION TO GET ALL HELPER VALUES

	def get_train_keys(self,data):
		self.standardize(data)

	# # HELPERS
	def standardize(self,data):
		self.train_keys['standardscaler']=StandardScaler()
		self.train_keys['standardscaler'].fit(np.reshape(data,[-1,data.shape[2]]))
		

	def save_train_keys(self,path):
		with open(path,'wb+') as f:
			pickle.dump(self.train_keys,f)

	def restore_train_keys(self,path):
		with open(path,'rb') as f:
			self.train_keys=pickle.load(f)

		print('restored train keys')

