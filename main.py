import time
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, AveragePooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import tensorflow_datasets as tfds  # pip install tensorflow-datasets
import tensorflow as tf
from PIL import Image
import logging
import numpy as np



def mnist_make_model(image_w: int, image_h: int):
   # Neural network model
   model = Sequential()
   model.add(Dense(1024, activation='relu', input_shape=(image_w*image_h,)))
   model.add(Dropout(0.2))  # rate 0.2 - set 20% of inputs to zero
   model.add(Dense(1024, activation='relu'))
   model.add(Dropout(0.2))
   model.add(Dense(10, activation='softmax'))
   model.compile(loss='categorical_crossentropy', optimizer=RMSprop(),
            metrics=['accuracy'])
   return model


def mnist_mlp_train(model):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
   # x_train: 60000x28x28 array, x_test: 10000x28x28 array
    image_size = x_train.shape[1]
    train_data = x_train.reshape(x_train.shape[0], image_size*image_size)
    test_data = x_test.reshape(x_test.shape[0], image_size*image_size)
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')
    train_data /= 255.0
    test_data /= 255.0
   # encode the labels - we have 10 output classes
   # 3 -> [0 0 0 1 0 0 0 0 0 0], 5 -> [0 0 0 0 0 1 0 0 0 0]
    num_classes = 10
    train_labels_cat = keras.utils.to_categorical(y_train, num_classes)
    test_labels_cat = keras.utils.to_categorical(y_test, num_classes)
    print("Training the network...")
    t_start = time.time()
   # Start training the network
    model.fit(train_data, train_labels_cat, epochs=8, batch_size=64,
    verbose=1, validation_data=(test_data, test_labels_cat))

def mlp_digits_predict(model, image_file):
    image_size = 28
    img = keras.preprocessing.image.load_img(image_file, target_size=(image_size, image_size), color_mode='grayscale')
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr/255.0
    img_arr = img_arr.reshape((1, image_size*image_size))
    result = model.predict_classes([img_arr])
    return result[0]
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.get_logger().setLevel(logging.ERROR)

model = mnist_make_model(image_w=28, image_h=28)
mnist_mlp_train(model)
model.save('mlp_digits_28x28.h5')
model = tf.keras.models.load_model('mlp_digits_28x28.h5')
print(mlp_digits_predict(model, 'digit_0.png'))
print(mlp_digits_predict(model, '4.jpg'))
print(mlp_digits_predict(model, '6.png'))
print(mlp_digits_predict(model, '2.png'))
print(mlp_digits_predict(model, '8.png'))
print(mlp_digits_predict(model, '9.png'))