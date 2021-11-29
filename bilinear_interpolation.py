import cv2
import math as m
import numpy as np
img=cv2.imread("./x5.bmp",0)

#--------------------------------------------Q3-------------------------------------		
def bilinearinterpolation(img,c):
	cv2.imshow("dog",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows() 

	print(c)
	
	data=np.asarray(img)
	M1=len(data)
	N1=len(data[0])
	M2=int(M1*c)
	N2=int(N1*c)

	fin_mat=np.ones((M2,N2))*-1

	for i in range(M1):
		for j in range(N1):
			fin_mat[int(i*c)][int(j*c)]=data[i][j]

	Mx=int(i*c)
	My=int(j*c)

	for i in range(Mx+1):
		for j in range(My+1):
			if fin_mat[i][j]==-1:
				x=int(i/c)
				y=int(j/c)

				if m.ceil(x)!=x:
					x1=m.floor(x)
					x2=m.ceil(x)
				else:
					if x==0:
						x1=0
						x2=1
					else:
						x1=x-1
						x2=x
				if m.ceil(y)!=y:
					y1=m.floor(y)
					y2=m.ceil(y)
				else:
					if y==0:
						y1=0
						y2=1
					else:
						y1=y1-1
						y2=y1

				x1=int(x1)
				y1=int(y1)
				x2=int(x2)
				y2=int(y2)

				X=[ [x1,y1,x1*y1,1] , [x1,y2,x1*y2,1] , [x2,y2,x2*y2,1] , [x2,y1,x2*y1,1]]
				Y=[ [data[x1][y1]] , [data[x1][y2]] ,[data[x2][y2]] ,[data[x2][y1]] ]
				
				A=np.dot(np.linalg.inv(X),Y)
				fin_mat[i][j]=np.dot(np.array([x,y,x*y,1]),A)

	

	for i in range(Mx+1):
		for j in range(My+1,len(fin_mat[0])):
			fin_mat[i][j]=fin_mat[i][j-1]

	for j in range(len(fin_mat[0])):
		for i in range(Mx+1,len(fin_mat)):
			fin_mat[i][j]=fin_mat[i-1][j]

	# print(fin_mat)
	new_mat=np.zeros((M2,N2),dtype='uint8')
	for i in range(M2):
		for j in range(N2):
			new_mat[i][j]=int(fin_mat[i][j])
	cv2.imshow("dog",new_mat)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

binaryinterpolation(img,0.5)

#-------------------------Q4---------------------
def rotation(angle):
	angle=np.radians(angle)
	rot=np.array([ [np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0],[0,0,1]  ])
	return rot

def translation(tx,ty):
	trans=np.array([ [1,0,tx],[0,1,ty],[0,0,1]])
	return trans

def scale(s):
	sc=np.array([[s,0,0],[0,s,0],[0,0,1]])
	return scale

def geometric_translation(img, transformation):
	cv2.imshow("dog",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	print(transformation) 

	rot=rotation(45)
	sc=scale(2)
	trans=translation(30,30)

	A1=np.dot(trans,rot)
	A2=np.dot(A1,sc)
	A=np.dot(A2,np.linalg.inv(trans))
	print(A)
#geometric_translation(img,2)