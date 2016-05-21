# biRNN
Bidirectional RNN [GRU + LSTM] for language identification using SSC

First run setup.py as "python setup.py install" to install the MFCC and other DCT based features

Convert your audio files using Audacity or any other sound editing tool to a 16-bit PCM Wav format

Add an "e" or an "s" to the filename at the beginning based on whether it is in English or Spanish 
e.g. Platero y Yo (spanish) -> "splateroyyo.wav"

run gen_features.py 

This will generate 2 set of features - 
	1. Signal features in the frequency domain divided into blocks
	2. MFCC features (or whichever you want from the 'features' folder)

Give these as input to you NN architecture.

For best results - Signal features to a ConvNet and MFCC directly to an LSTM or GRU

