#thuật toán giảm dần tốc độ

import numpy as np 
t = np.array([-1, -1, -1, 1, 1, 1])
x = np.array([ 	[1, 1,0,0,1, 0,1,1,0, 0,1,1,0, 1,0,0,1],
				[1, 0,0,0,0, 1,1,1,1, 1,1,1,1, 0,0,0,0],
				[1, 0,1,1,0, 0,1,1,0, 0,1,1,0, 0,1,1,0],
				[1, 1,1,1,1, 0,0,0,0, 0,0,0,0, 1,1,1,1],
				[1, 1,0,0,1, 1,0,0,1, 1,0,0,1, 1,0,0,1],
				[1, 1,0,0,0, 1,0,0,0, 1,0,0,0, 1,1,1,1]
	])

w =0.5-np.random.random((1,17))
print('Giá trị trọng số khởi tạo:')
print(w)
o=np.matmul(w,np.transpose(x))
print('Ngõ ra no-ron trước khi huấn luyện',np.sign(o)[0,:])
epoch=5
n=0.05
for i in range(0,epoch):
	dw=0
	for sample in range(0,6):
		o = np.matmul(x[sample,:],np.transpose(w))
		dw = dw + n*(t[sample]-o)*x[sample,:]
	w = w+ dw

	o = np.sign(np.matmul(w,np.transpose(x)))
	E = 0.5*np.sum(np.power((t-o),2))	
	print('Epoch',i,':',E)
o = np.matmul(w,np.transpose(x))
print('ngõ ra no-ron sau khi được huấn luyện ', np.sign(o)[0,:])

