import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
print (tf.__version__)

print ("Num of GPU available {}".format(len (tf.config.experimental.list_physical_devices('GPU'))))

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape (x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape (x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical (y_train, 10)
y_test = keras.utils.to_categorical (y_test, 10)

x_train = x_train.astype ('float32')
x_test = x_test.astype ('float32')
x_train /= 255
x_test /= 255

batch_size = 128
num_classes = 10
epochs = 10

model = keras.Sequential()
model.add (layers.Conv2D (32, kernel_size = (3,3),
                   activation = 'relu',
                   input_shape = input_shape))
model.add (layers.Conv2D (64, (3,3), activation = 'relu'))
model.add (layers.MaxPooling2D (pool_size = (2,2)))
model.add (layers.Dropout (0.25))
model.add (layers.Flatten ())
model.add (layers.Dense (256, activation = 'relu'))
model.add (layers.Dropout (0.5))
model.add (layers.Dense (num_classes, activation = 'softmax'))


model.compile (loss = keras.losses.categorical_crossentropy, 
               optimizer  = keras.optimizers.Adadelta(),
               metrics = ['accuracy'])
model.add (layers.Conv2D (32, kernel_size = (3,3),
                   activation = 'relu',
                   input_shape = input_shape))
model.add (layers.Conv2D (64, (3,3), activation = 'relu'))
model.add (layers.MaxPooling2D (pool_size = (2,2)))
model.add (layers.Dropout (0.25))
model.add (layers.Flatten ())
model.add (layers.Dense (256, activation = 'relu'))
model.add (layers.Dropout (0.5))
model.add (layers.Dense (num_classes, activation = 'softmax'))


model.compile (loss = keras.losses.categorical_crossentropy, 
               optimizer  = keras.optimizers.Adadelta(),
               metrics = ['accuracy'])
model = keras.Sequential()
model.add (layers.Conv2D (32, kernel_size = (3,3),
                   activation = 'relu',
                   input_shape = input_shape))
model.add (layers.Conv2D (64, (3,3), activation = 'relu'))
model.add (layers.MaxPooling2D (pool_size = (2,2)))
model.add (layers.Dropout (0.25))
model.add (layers.Flatten ())
model.add (layers.Dense (256, activation = 'relu'))
model.add (layers.Dropout (0.5))
model.add (layers.Dense (num_classes, activation = 'softmax'))


model.compile (loss = keras.losses.categorical_crossentropy, 
               optimizer  = keras.optimizers.Adadelta(),
               metrics = ['accuracy'])
