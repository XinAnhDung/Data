#Tải về các thư viện cần thiết
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tensorflow as tf
import seaborn as sns

#Ta tải thư viện mnist về hoặc từ đường dẫn
mnist = tf.keras.datasets.mnist
(X_train_all, Y_train_all), (X_test_all, Y_test_all) = mnist.load_data()
#Ta tiến hành kiếm tra dữ liệu của thư viện mnist 
print(Y_train_all.dtype, "-", Y_train_all.shape)
print(X_test_all.dtype, "-", X_test_all.shape)
print(Y_test_all.dtype, "-", Y_test_all.shape)
#Kiểm tra ngẫu nhiên dữ liệu tại một điểm bất kì
X_train_all[25]
Y_train_all[25]
#Thực hiện hiển thị dữ liệu
print("Load Mnist Database")
print(X_train_all.shape)
print(Y_train_all.shape)
print(X_test_all.shape)
print(Y_test_all.shape)
print(X_train_all[0,10:20,10:20])
print(Y_train_all[0])
X_train_all = np.reshape(X_train_all,(60000,784))/255.0
X_test_all = np.reshape(X_test_all,(10000,784))/255.0
Y_train_all = np.matrix(np.eye(10)[Y_train_all])
Y_test_all = np.matrix(np.eye(10)[Y_test_all])
def sigmoid(x):
	return 1./(1.+np.exp(-x))
def AccTest(outN,labels): #caculate the matching score
	outMaxArg = np.argmax(outN,axits=1)
	LabelMaxArg = np.argmax(labels,axits=1)
	Accuracy = np.mean(outMaxArg == LabelMaxArg)
	return Accuracy
def Forwardpass(sample,Wh,Wo):
	netH = np.dot(sample,Wh)
	oH = sigmoid(netH)
	netO = np.dot(oH,Wo)
	o = sigmoid(netO)
	return o

NumOfTrainSample = 60000
NumOfTestSample = 10000

NumInput = 784; # mnist_trani_sample.shape[1]
NumHidden = 512
NumOutput = 10; # number of classes

#Hidden layer

Wh = np.random.uniform(-0.5,0.5,(NumInput,NumHidden))

#output layer

Wo = np.random.uniform(-0.5,0.5,(NumHidden,NumOutput))

n = 0.1

for epoch in range(0,10):
	for sample in range(0,NumOfTrainSample):
		#forward pass

		x = X_train_all[sample,:]
		netH = np.dot(x,Wh)
		oH = sigmoid(netH)
		netO = np.dot(oH,Wo)
		o = sigmoid(netO)
		t = Y_train_all[sample,:]
		d = np.multiply(1-o,(t-o))
		d = np.multiply(d,o) #o(1-o)(t-o)
		dwo = np.dot(oH.T,d)
		#back propagate error
		dh = np.dot(Wo,d.T)
		dH = np.multiply((1-oH),dh.T) #o(1-o)(t-o)
		dH = np.multiply(oH,dH) #o(1-o)(t-o)
		dwh = np.dot(np.matrix(x).T,dH)
		#update weights
		Wh = Wh + n*dwh
		Wo = Wo + n*dwo
	o = Forwardpass(X_test_all,Wh,Wo)
	acc = AccTest(Y_test_all,np.matrix(o))
	print(acc)
	print(epoch)