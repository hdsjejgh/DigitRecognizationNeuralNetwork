import os #gets the images from the samples directory
import numpy as np #stores the image data in an array
import cv2 #reads the image
import matplotlib.pyplot as plt #shows the digits as an image
import tensorflow as tf #does all the hard stuff

"""
#actually trains the model
mnist = tf.keras.datasets.mnist #loads in the mnist database of letters
(x_train, y_train), (x_test, y_test) = mnist.load_data() #gets the training and testing data from the database and puts it into variables (x is pixel data, y is digits)

x_train = tf.keras.utils.normalize(x_train, axis=1) #normalizes the training and testing pixel data so that all the values are 0-1 (representing shade of grey)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#creates the model and adds the layers
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) #28x28 is the resolution of the images, it gets flattened into a 1x784 array
model.add(tf.keras.layers.Dense(128,activation='relu')) #honestly no clue what the relu layers do, the weights of the connections change in response to accuracy or something
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax')) #softmax layer converts outputs of all 10 neurons into probabilities to represent how sure the ai is of its answer

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train, y_train, epochs=40) #trains the model with the training data, goes over the numbers 40 times

model.save('mnist.keras') #saves the model

"""
model = tf.keras.models.load_model('mnist.keras') #loads model

image_number = 1 #the image numbers start at 1
while os.path.isfile(f"samples/digit{image_number}.png"): #repeats until there is no images left to try
    try:
        img = cv2.imread(f"samples/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img])) #makes the images white on black
        prediction = model.predict(img) #predicts the digit the image shows
        print(f"The Number is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show() #shows the image so you can see if the ai is right
    except:
        print("error") #incase the image cant be loaded
    finally:
        image_number+=1
