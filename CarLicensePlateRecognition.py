#library 
from PIL import Image
import numpy as np
import os
from os import listdir
import random
import math


#define activation function :
def Sigmoid(mat):
#        sigmoid1 = []
#        for i in range(len(mat)) :
#            for j in range(len(mat)):
#                k = mat[i][j]
#                sigmoid1 += [1/(1+(math.e ** (-k) ))]
#        return sigmoid1
     return 1/(1+math.e ** (-(mat)) )

def relu(mat):
    return np.maximum(0.0,mat)


#open :
FD = (r"C:\Users\parca\Desktop\proje gosste_phares 2 & 3\Train")
for folder in os.listdir(FD):
    FD = (r"C:\Users\parca\Desktop\proje gosste_phares 2 & 3\Train")
    FD = (r"C:\Users\parca\Desktop\proje gosste_phares 2 & 3\Train") + '\\' + folder
    for ima in os.listdir(FD) :
        if (ima.endswith(".png") or ima.endswith(".jpeg") or ima.endswith(".jpg")) or (ima.endswith(".jfif")):
            im = Image.open(FD+'\\'+ima)
            
            #resize and pixels number :
            im = im.resize((500,500))
            
            #convert black and white :
            im = im.convert('L')
            
            #array with numpy and create array of pixels between 0 <= x <= 1:
            twod_im = np.asarray(im)
            twod_im = twod_im.flatten()/255
            matris_twod = np.reshape(twod_im,(len(twod_im),1))
           
            #hidden layer :
            
            #randow weight :
            weight = []
            for i in range(10):
                tmp = []
                for j in range(len(matris_twod)):
                    tmp += [random.uniform(-1,1)]
                weight += [[tmp]]    
            
            #random bias :
            bias = []
            for i in range(10):
                bias += [[random.uniform(-1,1)]]

            #multiplication and sumation :
            #at first u should multipy mat and weight:
            nod_total = np.dot(weight , matris_twod)    
            #then u should add bias with nod_total :
            nod_total = np.add(bias , nod_total) 

            #activation func :
            nod_total = Sigmoid(nod_total)
            layer2_temp = nod_total

            #output layer :
            weight2 = []
            for i in range(28):
                tmp2 = []
                for j in range(10):
                    tmp2 += [random.uniform(-1,1)]
                weight2 += [[tmp2]]
            bias2 = []
            for i in range(28):
                tmp3 = []
                for j in range(10):
                    tmp3 += [random.uniform(-1,1)]
                bias2 += [[tmp3]] 
            #multiplication and sumation :
            #at first u should multipy mat and weight:
            nod_total = np.dot( weight2 , nod_total)    
            #then u should add bias with nod_total :
            nod_total = np.add(bias2 , nod_total) 

            #activation layer :
            nod_total = Sigmoid(nod_total)       
            

            #We want to determine which letter)(image) it is :
            name_of_pics_ph3 = ['0','1','2','3','4','5','6','7','8','9','A','B','D','Gh','H','J','L','M','N','P','PuV','PwD','Sad','Sin','Ta','Taxi','V','Y',]
            which_of_28 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            i_nameofpicsph3 = -1
            for x in name_of_pics_ph3 :
                if x in ima[:4]:
                    i_nameofpicsph3 += 1
                    which_of_28[i_nameofpicsph3] = 1
                    break
            nod_total[0][i_nameofpicsph3] -= which_of_28[i_nameofpicsph3]
            
            
            #output :
            #print(nod_total,len(nod_total))
            output1 = np.dot(layer2_temp[i_nameofpicsph3:][0].T,nod_total[0][i_nameofpicsph3])
            #print(nod_total)
            print(output1)