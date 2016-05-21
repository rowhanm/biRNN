import scipy.io.wavfile as wav
import numpy as np
import os
from features import mfcc

filename= "splateroyyo.wav"
data = wav.read(filename)
np_arr = data[1].astype('float32')/32767.0
bitrate = data[0]
data = np_arr
freq = 44100 #sampling freq, set as you please
block_size = freq/4
clip_len = 10 # check what value would be appropriate for this, GRUV kept this as 10 seconds
max_freq_len = int(round((freq*clip_len)/block_size)) 
block_list = []
total_samples = data.shape[0]
num_s = 0
while(num_s<total_samples):
	block = data[num_s:num_s+block_size]
	if(block.shape[0]<block_size):
		padding = np.zeros((block_size-block.shape[0]),)
		block = np.concatenate((block, padding))
	block_list.append(block)
	num_s += block_size	

fft_blocks = []
for block in block_list:
	fft_block = np.fft.fft(block)
	new_block = np.concatenate((np.real(fft_block), np.imag(fft_block)))
	fft_blocks.append(new_block)
	
X = fft_blocks

### this block of code ###
cur_seq = 0
chunk_X= []
tot_seq =len(X)
while (cur_seq+max_freq_len<tot_seq):
	chunk_X.append(X[cur_seq:cur_seq+max_freq_len])
	cur_seq+=max_freq_len
num_examples = len(chunk_X)

# check filename and assign Y as per that #
if filename[0] =='e':
	chunk_Y = np.zeros(num_examples) #or whatever as per the file	
elif filename[0] =='s':
	chunk_Y = np.ones(num_examples)
# here first letter of filename is either e or s  #
	
### not sure what this sequencing into 40 sequences does - possibly this creates a 3D array ###	
np.save("sig_feat_x",chunk_X)
np.save("sig_feat_y",chunk_Y)


#mfcc features - make changes as you want in the mfcc definition#
(rate,sig) = wav.read(filename)
mfcc_x = mfcc(sig,rate)
if filename[0] =='e':
	mfcc_y = np.zeros((len(mfcc_x)))
elif filename[0] =='s':
	mfcc_y = np.ones((len(mfcc_x)))

np.save("mfcc_x",mfcc_x)
np.save("mfcc_y",mfcc_y)
 	