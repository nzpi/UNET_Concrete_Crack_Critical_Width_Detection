from tensorflow.keras.layers import Input, BatchNormalization
from tensorflow.python.keras.engine.training import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np

IMAGE_SIZE = (256,256,1)

def model(weights_input=None):

    inputs = Input(IMAGE_SIZE)
    conv1 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(inputs)
    conv1 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv1)
    conv1 = BatchNormalization(axis=3)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool1)
    conv2 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv2)
    conv2 = BatchNormalization(axis=3)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool2)
    conv3 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv3)
    conv3 = BatchNormalization(axis=3)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool3)
    conv4 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv4)
    conv4 = BatchNormalization(axis=3)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool4)
    conv5 = Conv2D(1024, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv5)
    conv5 = BatchNormalization(axis=3)(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    up6 = Conv2D(512, 2, activation="relu", padding="same", kernel_initializer="he_normal")(UpSampling2D(size = (2,2))(drop5))
    merge6 = Concatenate(axis=3)([drop4,up6])
    conv6 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge6)
    conv6 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv6)
    conv6 = BatchNormalization(axis=3)(conv6)

    up7 = Conv2D(256, 2, activation="relu", padding="same", kernel_initializer="he_normal")(UpSampling2D(size = (2,2))(conv6))
    merge7 = Concatenate(axis=3)([conv3,up7])
    conv7 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge7)
    conv7 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv7)
    conv7 = BatchNormalization(axis=3)(conv7)


    up8 = Conv2D(128, 2, activation="relu", padding="same", kernel_initializer="he_normal")(UpSampling2D(size = (2,2))(conv7))
    merge8 = Concatenate(axis=3)([conv2,up8])
    conv8 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge8)
    conv8 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv8)
    conv8 = BatchNormalization(axis=3)(conv8)


    up9 = Conv2D(64, 2, activation="relu", padding="same", kernel_initializer="he_normal")(UpSampling2D(size = (2,2))(conv8))
    merge9 = Concatenate(axis=3)([conv1,up9])
    conv9 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge9)
    conv9 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv9)
    conv9 = Conv2D(2, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv9)
    conv9 = BatchNormalization(axis=3)(conv9)

    conv10 = Conv2D(1, 1, activation="sigmoid")(conv9)

    model = Model(inputs= [inputs], outputs=[conv10])

    def dice_coeff(y_true, y_pred, smooth=1):
        intersection = K.sum(y_true * y_pred, axis=[1,2,3])
        union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
        return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
    
    def dice_coef_loss(y_true, y_pred):
      return -dice_coeff(y_true, y_pred)


    optimizer = keras.optimizers.Adam(learning_rate=0.00001)


    model.compile(optimizer= optimizer, loss = "binary_crossentropy", metrics = ['accuracy',f1_m,precision_m, recall_m])

    if weights_input:
        model.load_weights(weights_input)

    return model

def prepare_input(image):
    image = np.reshape(image, image.shape+(1,))
    image = np.reshape(image,(1,)+image.shape)
    image = np.clip(image, 0, 255)
    return np.divide(image, 255)

def prepare_output(image):
    image = image[:,:,0]
    image = np.clip(image, 0, 1)
    return np.multiply(image, 255)


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))