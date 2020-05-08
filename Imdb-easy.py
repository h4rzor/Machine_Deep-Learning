from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D,MaxPooling1D
from keras.datasets import imdb
#Importing the necessary libraries

max_features = 20000
maxlen = 100
embedding_size=128
kernel_size = 5
filters = 64
pool_size = 4
lstm_output_size=70
batch_size = 32
epochs = 2
#These are the parameters that I will use to make the model

(X_train,y_train),(X_test,y_test) = imdb.load_data(num_words=max_features)
#We have the dataset already in the format that we need. It consist of sequences of words
#turn into integers. This is the format the model wants. It cannot process words.
X_train = sequence.pad_sequences(X_train,maxlen=maxlen)
X_test = sequence.pad_sequences(X_test,maxlen=maxlen)
#We are padding the integers sequences because not all sentences are equal. They all have to be
#equal in order the model to work. The sequences are padded with 0 because it does not
#bring anything to the table, but it accomplishes the task to transform the sequences 
#to be of equal length.


def build_model():
	model = Sequential()
	#The model is sequential. That means we will be adding layers in a sequential manner.
	model.add(Embedding(max_features,embedding_size,input_length=maxlen))
	#The embedding layer is used mostly in the htext processing part of deep learning.
	#It receives a sequence of integers and maps each integer(word) to a corresponding
	#vector. Then these vectors are each computed and we could see some of them close
	#to each other and other no so much -> the words are same or similar.
	model.add(Dropout(0.2))
	#The dropout layer is here to prevent overfiting. It does as the name suggests - 
	#drops some random nodes from the network.
	model.add(Conv1D(filters,kernel_size,padding='valid',activation='relu',strides=1))
	#The conv1d layer is the same as the commonly used conv2d in image classification tasks.
	#It differs from conv2d because it does the same but on the 1d tensor - the embedded sentence
	model.add(MaxPooling1D(pool_size=pool_size))
	#If we have three words and their corresdponding integers are 10,12,14
	#the maxpolling layer will chose the one with 14 -> this is why MAXpooing
	#It also is used to make the sequence smaller.
	model.add(LSTM(lstm_output_size))
	#This is a Long Short Term Memory cell. It is preferred before the classic RNN because 
	#it keeps the memory state throughout all the cells and prevents gradient explosion and undertraining 
	model.add(Dense(1))
	#This is the classic node.
	model.add(Activation('sigmoid'))
	#Sigmoid squashes the probability between 0 and 1
	return model
model = build_model()
#Building the model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#Compiling the model.For the loss parameter I used binary_crossentropy - because after all
#I am dealing with binary problem - whether the review is positive or negative
#The optimizer parameter is very important because it updates the weights 
#in such way that it is trying to minimize the error between predicted(y) and true(y)
model.fit(X_train,y_train,batch_size=batch_size,epochs=2)
#I am fitting the model with the forementioned X_train(which contains all the sentences) 
#and y_train(which contains all the labels for those sentences)
#I am running it for 2 epochs
#Setting the batch_size parameter
score,acc = model.evaluate(X_test,y_test,batch_size=batch_size)
#Evaluating the model with X_test and y_test
print(score,acc)
#The accuracy is not bad. Maybe if I add some more layers it will become more accurate?
#What do you think?
