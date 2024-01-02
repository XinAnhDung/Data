import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import datetime as dt 
print("Load Mnist Database")
mnist = tf.keras.datasets.minst 
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = np.reshape(x_train,(60000,784))/255.0
x_test= np.reshape(x_test,(60000,784))/255.0
y_train = np.matrix(np.eye(10)[y_train])
y_test = np.matrix(np.eye(10)[y_test])
x_train = np.matrix(x_train)
y_train = np.matrix(y_train)
x_test = np.matrix(x_test)
y_test = np.matrix(y_test)


def sigmol(x):
	return 1/(1+np.exp(-x))

def softmax(x):
	return np.divide(np.matrix(np.exp(x)),np.mat(np.sum(np.exp(x),axis=1)))

def AccTest(outN,labels):
	OutMaxArg = np.argmax(outN,axis=1)
	LabelMaxArg = np.argmax(labels,axis=1)
	Accuracy = np.mean(OutMaxArg==LabelMaxArg)
	return Accuracy

def feedforward(samples,Wh,bh,Wo,bo):
	bo = 0
	bh = 0
	OutH1 = sigmol(np.dot(samples,Wh)+bh)
	OutN = softmax(np.dot(OutH1,Wo)+bo)
	return OutN

learningRate = 0.1
learningRate_decay = 0.05
lamda = 0.0001
Anpha = 0.8
Batch_size = 100
Momentum = 0.95
Epoch = 10

NumOfTrainSample = 60000
NumOfTestSample = 10000

NumInput = 784
NumHidden = 512
NumOutput = 10


Wh = np.matrix(np.random.uniform(-0.5,0.5,(NumInput,NumHidden)))
bh= np.random.unifirm(0,0.5,(1,NumHidden))
del_Wh = np.zeros((NumInput,NumHidden))
del_bh = np.random((1,NumHidden))

Wo = np.random.uniform(-0.5,0.5,(NumHidden,NumOutput))
bo = np.random.uniform(0,0.5,(1,NumOutput))
del_Wo = np.zeros((NumHidden,NumOutput))
del_bo = np.zeros((1,NumOutput))

SampleIdx = np.concatenate([np.arange(NumOfTrainSample),np.arange(NumOfTrainSample)])
Cost_Entropy = np.zeros(Epoch)
Cost = np.zeros(np.int(np.ceil(NumOfTrainSample/Batch_size)))
IdxCost = 0;
t_start = t1 = dt.datetime.now()
for ep in range(Epoch):
	t1 = dt.datetime.now()
	Idx = np.random.randint(0,NumOfTrainSample,1)
	for i in range(0,NumOfTrainSample,Batch_size):
		Batch_size = SampleIdx[i + Idx[0]:i + Idx[0] + Batch_size]
		sample = np.matrix(x_train[Batch_size,:])
		targ = np.matrix(y_train[Batch_size,:])
		bh = 0
		bo = 0
		OutH1 = sigmol(np.dot(sample,Wh)+bh)
		OutN = softmax(np.dot(OutH1,Wo)+bo)
		Cost[IdxCost] =- np.sum(np.multiply(targ,np.log10(OutN)))
		IdxCost+=1
		ho = (targ - OutN)

		dWo = np.matrix(np.dot(OutH1.T,ho)/Batch_size)
		dbo = np.mean(ho,0)

		temp = np.dot(WO,ho.T)
		temp1 = np.multiply(OutH1,(1 - OutH1))
		hH1 = np.multiply(temp.T,temp1)

		dwh1 = np.dot(sample.T,hH1)/Batch_size
		dbh1 = np.mean(hH1,0)

		WoUpdate = learningRate*dWo1 + Momentum*del_Wo
		boUpdate = learningRate*dbo + Momentum*del_bo
		del_Wo = WoUpdate
		del_bo = boUpdate

		WhUpdate = learningRate*dwh1 + Momentum*del_Wh
		bhUpdate = learningRate*dbh1 + Momentum*del_bh
		del_Wh = WhUpdate
		del_bh = bhUpdate

		Wo = Wo + WoUpdate
		bo += boUpdate
		Wh = Wh + WhUpdate
		bh += bhUpdate
	Cost_Entropy[ep] = np.mean(Cost)
	IdxCost = 0
	learningRate = learningRate_decay
	t2 = dt.datetime.now()
	print(t2-t1)
	print("Tring epoch %i" %ep)
	print("Cross-Entropy %f" %Cost_Entropy[ep])
	RealOutN = feedforward(x_test,Wh,bh,Wo,bo)
	Accuracy = AccTest(RealOutN,y_test)
	print("Accuracy: %f", %Accuracy )
t_end = t1 =dt.datetime.now()
print("Total time: ")
print(t_end-t_start)
plt.plot(Cost_Entropy,"dr-")
plt.show()