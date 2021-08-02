#Import all libraries
import os
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# =============================================================================
# #Getting data
# os.chdir("..")
# os.chdir("..")
# os.chdir("R/Projects/Project2/Face Mask Dataset/Train/WithMask")
# print(os.getcwd())
# =============================================================================

x = os.listdir()
train =[]
for i in range(0,len(x)) :
    train.append(np.array(Image.open(x[i])))

os.chdir("..")
os.chdir("WithoutMask")
x = os.listdir()
for i in range(0,len(x)) :
    train.append(np.array(Image.open(x[i])))    

os.chdir("..")
os.chdir("..")
os.chdir("Test")
os.chdir("WithoutMask")
x = os.listdir()
test1 =[]
for i in range(0,len(x)) :
    test1.append(np.array(Image.open(x[i])))
    
os.chdir("..")
os.chdir("WithMask")
x = os.listdir()
for i in range(0,len(x)) :
    test1.append(np.array(Image.open(x[i])))
    

#data Y Labels
#With Mask = 1 and WIth out = 0
trainy=[]
for i in range(0,5000):
    trainy.append(1)
for i in range(5000,10000):
    trainy.append(0)
    
testy=[]
for i in range(0,509):
    testy.append(0)
for i in range(509,992):
    testy.append(1)

#Resize and Reshape
for i in range(0,len(train)):
    train[i] = np.resize(train[i],(64,64,3))
    train[i] = np.reshape(train[i],(64,64,3))

for i in range(0,len(test1)):
    test1[i] = np.resize(test1[i],(64,64,3))
    test1[i] = np.reshape(test1[i],(64,64,3))

#Reshaping
for i in range(0,len(train)):
    train[i] = np.reshape(train[i],(-1,64*64*3))

for i in range(0,len(test1)):
    test1[i] = np.reshape(test1[i],(-1,64*64*3))

train=np.array(train).reshape(10000,64*64*3)
test1=np.array(test1).reshape(992,64*64*3)

##
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)

values=np.array(trainy)
integer_encoded = label_encoder.fit_transform(values)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y = onehot_encoder.fit_transform(integer_encoded)

values=np.array(testy) 
integer_encoded = label_encoder.fit_transform(values)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
testy = onehot_encoder.fit_transform(integer_encoded)
    
#Model Define
model = Sequential()
model.add(Dense(256,input_dim=12288, activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train,y,epochs=50,batch_size=250)

model.evaluate(train,y)


def predict(filename):
    Im = Image.open(filename)
    z=np.array(Im)
    z=np.resize(z,(64,64,3))
    z=np.reshape(z,(-1,64*64*3))
    p=model.predict(z)
    if p[0][0] > p[0][1]:
        return 'Without Mask'
    else:
        return 'With Mask'
    
    
    

    
    
    