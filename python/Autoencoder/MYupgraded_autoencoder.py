import keras
import numpy as np
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from include_file import load_data
from sklearn.model_selection import train_test_split

# loading data
(train_data, train_labels), (test_data, test_labels) = load_data()

# Shapes of training set
print("Training set (images) shape: {shape}".format(shape=train_data.shape))

# Shapes of test set
print("Test set (images) shape: {shape}".format(shape=test_data.shape))

# Create dictionary of target classes
label_dict = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
}

# pre-process data to feed into network
train_data = train_data.reshape(-1, 28, 28, 1)
test_data = test_data.reshape(-1, 28, 28, 1)
print(train_data.shape, test_data.shape)

# re-scale data to max value of 1.0
train_data = train_data / np.max(train_data)
test_data = test_data / np.max(test_data)

# split training data into 2 parts (pretraining and validation?)
train_X, valid_X, train_ground, valid_ground = train_test_split(train_data,
                                                                train_data,
                                                                test_size=0.2,
                                                                random_state=13)

# set model parameters
batch_size = 64
epochs = 1
inChannel = 1
x, y = 28, 28
input_img = Input(shape=(x, y, inChannel))
num_classes = 10

# Change the labels from categorical to one-hot encoding, in this case row vectors 1x10
train_Y_one_hot = to_categorical(train_labels)
test_Y_one_hot = to_categorical(test_labels)

# split data using the new one-hot encoding
train_X, valid_X, train_label, valid_label = train_test_split(train_data, train_Y_one_hot, test_size=0.2,
                                                              random_state=13)

# define model with decoder removed
def encoder(input_img):
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4


# add full connected layers
def fc(enco):
    flat = Flatten()(enco)
    den = Dense(128, activation='relu')(flat)
    out = Dense(num_classes, activation='softmax')(den)
    return out


# put together classification model
encode = encoder(input_img)
full_model = Model(input_img, fc(encode))

# compile the model
full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
full_model.summary()

# run model
classify_train = full_model.fit(train_X, train_label, batch_size=64, epochs=3, verbose=1,
                                validation_data=(valid_X, valid_label))

full_model.save_weights('autoencoder_classification.h5')


for layer in full_model.layers[0:19]:
    layer.trainable = True
# compile fine-tuning
full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# run fine-tuning
classify_train = full_model.fit(train_X, train_label, batch_size=64, epochs=3, verbose=1,
                                validation_data=(valid_X, valid_label))

full_model.save_weights('classification_complete.h5')

# classification and evaluation on test data
test_eval = full_model.evaluate(test_data, test_Y_one_hot, verbose=0)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

predicted_classes = full_model.predict(test_data)

predicted_classes = np.argmax(np.round(predicted_classes), axis=1)

print(predicted_classes.shape, test_labels.shape)

correct = np.where(predicted_classes == test_labels)[0]
print "Found %d correct labels" % len(correct)

incorrect = np.where(predicted_classes != test_labels)[0]
print "Found %d incorrect labels" % len(incorrect)

from sklearn.metrics import classification_report

target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_labels, predicted_classes, target_names=target_names))

print("done")
