---
title: Traffic signs classification with Deep convolutional Neural Networks
featured: images/blog/2017-07-19-Traffic-signs-classification.jpg
layout: post
date: '2016-05-29'
category: [Data science, Deep learning]
tags: [Data science, Deep learning]
author: luis
---

In this post, I will face traffic signs classification problem with a convolutional neural network implemented with Keras on top of TensorFlow. Some background in neural networks is expected and I won't explain too much about how they work.

The dataset used to validate our algorithm is the well-known and widely used German Traffic Sign Dataset that consists of **39,209 32×32 px color images** for training and **12,630** images for testing. Each image is represented by an 32×32×3 array of pixels which contains RGB color values between 0 and 255. There are 43 traffic sign types. As observerd in the picture below, classes are very unbalanced and there are some classes much better represented that others. This is an important factor to have in mind when designing our classification model.

![alt text](/assets/images/blog/class_balancing.png "Class balancing")

There are various aspects to consider when thinking about this problem:

* Play around preprocessing techniques (normalization, rgb to grayscale, etc)
* Generate fake data.
* Neural network architecture
* Number of examples per label (some have more than others)

As for preprocessing stage, pictures are converted from RGB to YUV. Only the channel Y is used. Pierre Sermanet and Yann LeCun showed in [this paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) that using color channels didn’t seem to improve things a lot. The constrast is adjusted by means of historgram. This alleviates the problem of having sample with really poor image contrast. Finally, for better performance while optimizing our neural network, each image is centered on zero mean and divided for its standard deviation.

```python
import cv2

def preprocess_features(X, equalize_hist=True):

    # convert from RGB to YUV
    X = np.array([np.expand_dims(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YUV)[:, :, 0], 2) for rgb_img in X])

    # adjust image contrast
    if equalize_hist:
        X = np.array([np.expand_dims(cv2.equalizeHist(np.uint8(img)), 2) for img in X])

    X = np.float32(X)

    # standardize features
    X -= np.mean(X, axis=0)
    X /= (np.std(X, axis=0) + np.finfo('float32').eps)

    return X

X_train_norm = preprocess_features(X_train)
X_test_norm = preprocess_features(X_test)
```
<br>
Given our data model will have a lot of parameters and dataset may not be sufficient for a model to generalize well, we will use data augmentation to increase the number of training samples. There are many ways to perform data augmentation. Here we do not reinvent the wheel and we benefit from the class `ImageDataGenerator` to rotate, shift and flip pictures. See Keras documentation on [ImageDataGenerator](https://keras.io/preprocessing/image/) for further information.

```python
from keras.preprocessing.image import ImageDataGenerator

image_datagen = ImageDataGenerator(rotation_range=15.0,
                                   zoom_range=0.2,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True,
                                   vertical_flip=True)
```
<br>
Next step is to write the model using Keras. We will build a simple model consisting of 3 convolutional layers and 1 fully connected layer. Categorical crossentropy is used a loss to optimize the weights with adam optimizer. We included `dropout` to reduce the overfitting. However, we have not experience much improvement.

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(32, 32, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(n_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', 'categorical_accuracy', fscore])
```
<br>
Below `categorical crossentropy` on training and validation datasets are shown after training the model over 160 epochs. The number of samples for each minibatch is 128.

![alt text](/assets/images/blog/training-validation.png "Training performance")

Running the trained model on the test dataset, we observe that it does a pretty good job. **Kappa score** is about **0.95**. Not too bad for a pretty simple model. Kappa is a robust performance metric for unbalanced problems. Accuracy is not actually a good metric. Pretty naive models could show high accuracy while misclassifying less frequent classes. Also, we visualize the confusion matrix to get a better taste about performance for each class.

![alt text](/assets/images/blog/confusion_matrix.png "Confusion matrix")
