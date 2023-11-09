# Skill Assisessment-Handwritten Digit Recognition using MLP
## Aim:
To Recognize the Handwritten Digits using Multilayer perceptron.
##  EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook
## Theory:
Introduction: The "Digit Recognition using Artificial Neural Networks (ANN)" project aims to create an advanced system capable of recognizing and classifying handwritten digits. By leveraging the power of machine learning, specifically Artificial Neural Networks, the project endeavors to accurately identify digits ranging from 0 to 9.

Dataset: The project utilizes the widely recognized MNIST dataset, a staple in the machine learning community. Comprising a collection of 28x28 grayscale images of handwritten digits, the dataset also includes corresponding labels, making it an ideal resource for training and testing the neural network.

Artificial Neural Network (ANN): The architecture of the Artificial Neural Network comprises multiple layers, including the input layer, hidden layers, and the output layer. By employing a combination of feedforward and backpropagation techniques, the network is designed to learn the intricate patterns and nuances within the dataset.

Conclusion: In summary, the "Digit Recognition using Artificial Neural Networks (ANN)" project showcases the prowess of deep learning in accurately classifying handwritten digits. By demonstrating the application of ANN in image recognition tasks, the project lays the foundation for further exploration and advancement in the field of computer vision and deep learning.

## Algorithm :

Load the MNIST dataset containing handwritten digit images and labels.
Preprocess the dataset:

Normalize the pixel values of the images to a suitable range.
Format the labels to prepare them for training.

Design the architecture of the Artificial Neural Network:
Define the number of layers, neurons in each layer, and the activation functions.

Initialize the weights and biases for the neural network.
Set the hyperparameters for training the model:
Define the learning rate, number of epochs, and batch size.

Train the Artificial Neural Network:
Iterate through the training data for the specified number of epochs.

Implement the feedforward mechanism to propagate input data through the network.
Utilize backpropagation to update the weights and biases, minimizing the loss function.

Evaluate the trained model:
Use the test dataset to assess the model's performance.
Calculate metrics such as accuracy, precision, recall, and the F1 score.

Deploy the model:
Create a user-friendly interface for users to input their own handwritten digits.
Implement the functionality to visualize the model's predictions in real time.
Conclude the project, highlighting the success of the ANN in accurately recognizing and classifying handwritten digits.


## Program:

```
DEVELOPED BY: SOUNDARIYAN M.N

REG:212222230146
```


```
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout,Conv2D,MaxPooling2D
from tensorflow.keras.models import load_model
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
x_train[0].shape
x_train[0]
plt.matshow(x_train[7])
y_train[7]
x_train_flattened=x_train.reshape(len(x_train),28*28)
x_test_flattened=x_test.reshape(len(x_test),28*28)
model = Sequential()
model.add(Conv2D(32,(3,3), input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(10,activation="softmax"))
model.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
model.summary()
f=model.fit(x_train,y_train,epochs=5, validation_split=0.3)
f.history
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(f.history['loss'], color = 'green', label='loss')
plt.plot(f.history['val_loss'], color = 'orange', label = 'val_loss')
fig.suptitle('LOSS', fontsize=20)
plt.legend(loc='upper left')
plt.show()
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(f.history['accuracy'], color = 'green', label='accuracy')
plt.plot(f.history['val_accuracy'], color = 'orange', label = 'val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='upper left')
plt.show()
prediction = model.predict(x_test)
print(prediction)
print(np.argmax(prediction[0]))
plt.imshow(x_test[0])
model.save(os.path.join('model','digit_recognizer.keras'),save_format = 'keras')
img = cv2.imread('test.png')
plt.imshow(img)
rimg=cv2.resize(img,(28,28))
plt.imshow(rimg)
new_model = load_model(os.path.join('model','digit_recognizer.keras'))
new_img = tf.keras.utils.normalize(rimg, axis = 1)
new_img = np.array(rimg).reshape(-1,28,28,1)
prediction = model.predict(new_img)
print(np.argmax(prediction))
new_img.shape

```



## Output :


## Result:
