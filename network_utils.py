from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.preprocessing import sequence
from keras.utils.np_utils import accuracy
from keras.layers import merge
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
#from keras.layers.convolutional import Convolutional1D, AveragePooling3D

def create_lstm_network(num_frequency_dimensions, num_hidden_dimensions, num_recurrent_units=1):
	model = Sequential()
	#This layer converts frequency space to hidden space
	model.add(TimeDistributedDense(input_dim=num_frequency_dimensions, output_dim=num_hidden_dimensions))
	for cur_unit in xrange(num_recurrent_units):
		model.add(LSTM(input_dim=num_hidden_dimensions, output_dim=num_hidden_dimensions, return_sequences=True))
		model.add(GRU(input_dim=num_hidden_dimensions, output_dim=num_hidden_dimensions, return_sequences=True, go_backwards=True))
	#This layer converts hidden space back to frequency space
	model.add(TimeDistributedDense(input_dim=num_hidden_dimensions, output_dim=num_frequency_dimensions))
	model.compile(loss='mean_squared_error', optimizer='rmsprop')
	return model

def create_gru_network(num_frequency_dimensions, num_hidden_dimensions, num_recurrent_units=1):
	model = Sequential()
	#This layer converts frequency space to hidden space
	model.add(TimeDistributedDense(input_dim=num_frequency_dimensions, output_dim=num_hidden_dimensions))
	for cur_unit in xrange(num_recurrent_units):
		model.add(GRU(input_dim=num_hidden_dimensions, output_dim=num_hidden_dimensions, return_sequences=True))
	#This layer converts hidden space back to frequency space
	model.add(TimeDistributedDense(input_dim=num_hidden_dimensions, output_dim=num_frequency_dimensions))
	model.compile(loss='mean_squared_error', optimizer='rmsprop')
	return model

def create_cxlygz(num_frequency_dimensions, num_hidden_dimensions,timedistributedinput):
	rnn_output_size = 1024
	left_branch = Sequential()
	#(samples, rows, cols, channels) for input dimensions of convolutional2D
	#channels= num_freq_dim, samples=67(put some vairbale for it),
	left_branch.add(TimeDistributed(Convolutional2D(100,40,num_frequency_dimensions, border_mode='same',dim_ordering='tf',activation='relu'),input_shape=(67,40,num_frequency_dimensions)))
	#(samples, new_rows, new_cols, nb_filter) output shape of convolutional
	#(samples, rows, cols, channels) input for maxpooling
	left_branch.add(MaxPooling2D(pool_size=(2,2),dim_ordering='tf'))
	#(samples, pooled_rows, pooled_cols, channels) output for maxpooling
	left_branch.add(LSTM(rnn_output_size), name='forward',input='MaxPooling2D')
	left_branch.add(GRU(rnn_output_size, go_backwards=True),name='backward',input='MaxPooling2D')
	left_branch.add(Dense(1), inputs=['forward', 'backward'])

	right_branch = Sequential()
	right_branch.add(TimeDistributedDense(input_dim=timedistributedinput,output_dim=num_hidden_dimensions)
	right_branch.add(LSTM(rnn_output_size),name='forward',input='TimeDistributedDense')
	right_branch.add(GRU(rnn_output_size,go_backwards=True),name='backward',input='TimeDistributedDense')
	right_branch.add(Dense(1), inputs=['forward','backward'])

	merged = Merge([left_branch, right_branch], mode='concat')

	final_model = Sequential()
	final_model.add(merged)
	final_model.add(Dense(1, activation='tanh'))

	final_model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
	print(model.outputs)
	return model

