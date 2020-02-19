
import numpy as np 
from tensorflow.keras.layers import Input,Dense,LSTM,RepeatVector,TimeDistributed
from tensorflow.keras import Model


def build_model(build_keys):
	if build_keys['mode']=='ae_rse':
		visible=Input(shape=build_keys['input_shape'])
		lstm1=LSTM(32,activation='relu',return_sequences=True)(visible)
		lstm2=LSTM(16, activation='relu', return_sequences=False)(lstm1)
		repeat=RepeatVector(build_keys['timesteps'])(lstm2)
		lstm3=LSTM(16, activation='relu', return_sequences=True)(repeat)
		lstm4=LSTM(32,activation='relu',return_sequences=True)(lstm3)
		outputs=TimeDistributed(Dense(build_keys['input_shape'][1]))(lstm4)
		model=Model(inputs=visible,outputs=outputs)

	elif build_keys['mode']=='rse':
		visible=Input(shape=build_keys['input_shape'])
		lstm1=LSTM(32,activation='relu',return_sequences=True)(visible)
		lstm2=LSTM(16, activation='relu', return_sequences=True)(lstm1)
		lstm3=LSTM(16, activation='relu', return_sequences=True)(lstm2)
		lstm4=LSTM(32,activation='relu',return_sequences=True)(lstm3)
		outputs=TimeDistributed(Dense(build_keys['input_shape'][1]))(lstm4)
		model=Model(inputs=visible,outputs=outputs)

	return model

