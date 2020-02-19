from tensorflow.keras.models import load_model
import pickle
import numpy as np
import os
import pandas as pd
import itertools
import matplotlib.pyplot as plt

def make_temporal_features(X,y,timesteps):
	temporal_X=[]
	temporal_y=[]
	for x in range(X.shape[0]-timesteps+1):
		temporal_X.append(X[x:x+timesteps])
		temporal_y.append(y[x+timesteps-1])
	return np.asarray(temporal_X),np.asarray(temporal_y)

def mse(A,B):
	a=np.subtract(A,B)
	b=np.square(a)
	c=np.mean(b,axis=(1,2))
	return c


def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,normalize=False):


    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

def plot_linear(list_):

	for x,y in zip(list_,[1,2,3,4]):
		plt.scatter(x,np.ones_like(x)*y)

	plt.show()


class test_model:
	def __init__(self,PATH,model_path,threshold_path,utils_path):
		self.model=load_model(model_path)
		with open(utils_path,'rb') as f:
			self.train_keys=pickle.load(f)
		with open(threshold_path,'rb') as f:
			self.threshold=pickle.load(f)


		### TUNE YOUR OWN THRESHOLD

		if mode=='ae_rse':
			self.threshold=0.155
		else:
			self.threshold=0.175

		shift=2
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

		self.X,self.y=make_temporal_features(X,y,timesteps=self.train_keys['timesteps'])
		print("threshold value ",self.threshold)

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
			yield return_batch_


	def predict(self):
		self.out_class=np.asarray([])
		self.loss_all=np.asarray([])
		i=0
		print("shape ",self.X.shape)
		gen=self.single_yielder(data=self.X,batch_size=2048)
		while True:
			try:
				X=next(gen)
			except:
				break
			y=self.model.predict(X)
			loss=mse(X,y)
			self.loss_all=np.hstack((self.loss_all,loss))
			out_class_batch=np.zeros_like(loss)
			out_class_batch[loss>self.threshold]=1
			self.out_class=np.hstack((self.out_class,out_class_batch))
			i+=1
		return self.out_class

	def accuracy(self,y_pred):
		from sklearn.metrics import roc_auc_score,accuracy_score
		print("roc_auc_score ",roc_auc_score(self.y,y_pred))
		from sklearn.metrics import confusion_matrix
		self.conf_mat=confusion_matrix(self.y, y_pred)


	def plot(self):
		plot_confusion_matrix(self.conf_mat,['normal','break'])
		# plot_linear(,)


if __name__=='__main__':

	mode='rse' ##  best threshold_rse=0.175   threshold_ae_rse=0.165
	output_data=os.path.join('..','output_data_{}'.format(mode))
	model_path=os.path.join(output_data,'h5_file','model.h5')
	utils_path=os.path.join(output_data,'train_keys.pkl')
	threshold_path=os.path.join(output_data,'threshold.pkl')
	data_path=os.path.join('..','data','rare_events.csv')

	model=test_model(PATH=data_path,model_path=model_path,utils_path=utils_path,threshold_path=threshold_path)
	
	
	y=model.predict()
	model.accuracy(y)
	model.plot()

