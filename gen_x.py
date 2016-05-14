from features import ssc
import scipy.io.wavfile as wav
file_name = "splateroyyo.wav"
(rate,sig) = wav.read(file_name)  #or whatever the filename is
f = ssc(rate,sig)
print f.shape

fd = open("x_train.npy","a+b")
np.save(fd,f)
fd.close()

if file_name[0] =='s':
	y_train = np.ones(f.shape[0],1)
elif file_name[0] =='e':
	y_train=np.zeros(f.shape[0],1)

fdd = open("y_train.npy","a+b")
np.save(fdd,y_train)
fdd.close()		

