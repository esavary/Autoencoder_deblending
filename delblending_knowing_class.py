import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D

from tensorflow.keras import backend as K
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.layers import Flatten, Reshape, Input
from tensorflow.keras.layers import Dense, Activation, Add
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from  tensorflow.keras.layers import concatenate
import glob
import os
import sys
from tensorflow.keras.optimizers import Adam, RMSprop
from astropy.io import fits as pyfits
from astropy.wcs import WCS
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l1

from tensorflow.keras.models import Model
from math import ceil, log

def difference_mag(n1,n2):

    exp1 = K.log(n1)
    exp1 = tf.ceil(exp1)
    exp2 = K.log(n2)
    exp2 = tf.ceil(exp2)
    return 10 ** (exp1-exp2)


def extract_data(path_lenses,path_non_lenses,path_substracted,path_LRG):
    #extract data data
    X=[]
    Y1=[]
    Y2 = []
    labels=[]

    list_of_files = sorted([os.path.basename(x) for x in glob.glob(path_lenses + '*.fits')])
    list_of_nonlenses = sorted([os.path.basename(x) for x in glob.glob(path_non_lenses + '*.fits')])

    i=0
    for name in list_of_files:

        try:
            lens = pyfits.open(path_lenses + name)[0].data
            sub = pyfits.open(path_substracted + name)[0].data
            LRG= pyfits.open(path_LRG + name)[0].data

            if np.shape(lens)==(44,44):

                X.append(lens)
                Y1.append(LRG)
                Y2.append(sub)
                labels.append([1])
                i = i + 1
        except FileNotFoundError:
            print('file not found')
    print(np.shape(X),np.shape(labels))
    for name in list_of_nonlenses:
        try:
            lens = pyfits.open(path_non_lenses + name)[0].data
            sub = np.zeros((44,44))
            LRG = lens
            if np.shape(lens) == (44, 44):
                X.append(lens)
                Y1.append(LRG)
                Y2.append(sub)
                labels.append([0])
                i = i + 1
        except FileNotFoundError:
            print('file not found')
    print(np.shape(X), np.shape(labels))
    return np.array(X),np.array(Y1),np.array(Y2),np.array(labels)



def ae_classification_from_latent_space():
    '''This model takes a lens candidate as input and outputs the foreground object image (LRG), the lensing features (source), the lens
    image (Addition of the LRG and the lensed source) and a classifition for the lens candidates.
    The classification is done from the features of the latent space.
    '''
    weights_name='classifier_autoencoder_heactivation.h5'
    img_rows, img_cols, img_chns =44, 44, 1
    latent_dim = 64
    intermediate_dim = 256
    epsilon_std = 1.0
    filters = 64
    num_conv = 3
    batch_size = 1

    #encoder
    encoder_inputs = Input(shape=(img_rows, img_cols, img_chns))
    x = Conv2D(img_chns,kernel_size=(2, 2),padding='same', activation='relu',kernel_initializer = 'he_normal')(encoder_inputs)
    pool2 = MaxPooling2D((2, 2), padding='same')(x)
    x1 = Conv2D(22,kernel_size=(2, 2),padding='same', activation='relu',kernel_initializer = 'he_normal',strides=(1, 1))(pool2)
    pool1 = MaxPooling2D((2, 2), padding='same')(x1)
    x4 = Flatten()(pool1)
    # create LRG output

    output_shape=(batch_size,11,11,22)

    reshape1= Reshape(output_shape[1:])(x4)
    up1 = UpSampling2D((2, 2))(reshape1)
    l1=Conv2DTranspose(filters,kernel_size=num_conv,padding='same',strides=1,kernel_initializer = 'he_normal',activation='relu')(up1)
    up2 = UpSampling2D((2, 2))(l1)
    outputLRG=Conv2DTranspose(img_chns,kernel_size=2,padding='same',activation='sigmoid')(up2)

    # Source output

    reshape2= Reshape(output_shape[1:])(x4)
    up_1 = UpSampling2D((2, 2))(reshape2)
    s1=Conv2DTranspose(filters,kernel_size=num_conv,padding='same',strides=1,kernel_initializer = 'he_normal',activation='relu')(up_1)
    up_2 = UpSampling2D((2, 2))(s1)
    outputsource=Conv2DTranspose(img_chns,kernel_size=2,padding='same',activation='sigmoid')(up_2)
    addlayer= Add()([outputLRG, outputsource])

    #classification output

    denseclass2=Dense(32,activation='relu',kernel_initializer = 'he_normal')(x4)
    dropout=Dropout(0.1)(denseclass2)
    denseclass3=Dense(1,activation='sigmoid')(dropout)


    #final model
    model = Model(encoder_inputs, [outputLRG,outputsource,addlayer,denseclass3])
    return model,weights_name


def ae_classification_from_source_image():
    '''This model takes a lens candidate as input and outputs the foreground object image (LRG), the lensing features (source), the lens
       image (Addition of the LRG and the lensed source) and a classifition for the lens candidates.
       The classification is done from the features of the lensed source image.
       '''
    weights_name='classifier_autoencoder_endclassification3.h5'
    img_rows, img_cols, img_chns =44, 44, 1
    filters = 64
    num_conv = 3
    batch_size = 1

    #encoder
    encoder_inputs = Input(shape=(img_rows, img_cols, img_chns))
    x = Conv2D(img_chns,kernel_size=(2, 2),padding='same', activation='relu',kernel_initializer = 'he_normal')(encoder_inputs)
    pool2 = MaxPooling2D((2, 2), padding='same')(x)
    x1 = Conv2D(22,kernel_size=(2, 2),padding='same', activation='relu',kernel_initializer = 'he_normal',strides=(1, 1))(pool2)
    pool1 = MaxPooling2D((2, 2), padding='same')(x1)
    x4 = Flatten()(pool1)
    # create LRG output


    output_shape=(batch_size,11,11,22)

    reshape1= Reshape(output_shape[1:])(x4)
    up1 = UpSampling2D((2, 2))(reshape1)
    l1=Conv2DTranspose(filters,kernel_size=num_conv,padding='same',strides=1,kernel_initializer = 'he_normal',activation='relu')(up1)
    up2 = UpSampling2D((2, 2))(l1)
    outputLRG=Conv2DTranspose(img_chns,kernel_size=2,padding='same',activation='sigmoid')(up2)

    # Source output

    reshape2= Reshape(output_shape[1:])(x4)
    up_1 = UpSampling2D((2, 2))(reshape2)
    s1=Conv2DTranspose(filters,kernel_size=num_conv,padding='same',strides=1,kernel_initializer = 'he_normal',activation='relu')(up_1)
    up_2 = UpSampling2D((2, 2))(s1)
    outputsource=Conv2DTranspose(img_chns,kernel_size=2,padding='same',activation='sigmoid')(up_2)
    addlayer= Add()([outputLRG, outputsource])

    #classification output
    x30= Flatten()(outputsource)
    dropout1 = Dropout(0.1)(x30)
    denseclass2=Dense(100,activation='relu',kernel_initializer = 'he_normal')(dropout1)
    dropout=Dropout(0.1)(denseclass2)
    denseclass3=Dense(1,activation='sigmoid')(dropout)


    #final model
    model = Model(encoder_inputs, [outputLRG,outputsource,addlayer,denseclass3])
    return model,weights_name


def ae_deblending_only():
    '''This model takes a lens candidate as input and outputs the foreground object image (LRG), the lensing features (source), the lens
          image (Addition of the LRG and the lensed source).

          '''
    weights_name='classifier_onlyencoder.h5'
    img_rows, img_cols, img_chns =44, 44, 1
    latent_dim = 64
    intermediate_dim = 256
    epsilon_std = 1.0
    filters = 64
    num_conv = 3
    batch_size = 1

    #encoder
    encoder_inputs = Input(shape=(img_rows, img_cols, img_chns),name='encoderinput')
    x = Conv2D(img_chns,kernel_size=(2, 2),padding='same', activation='relu',kernel_initializer = 'he_normal')(encoder_inputs)
    pool2 = MaxPooling2D((2, 2), padding='same')(x)
    x1 = Conv2D(22,kernel_size=(2, 2),padding='same', activation='relu',kernel_initializer = 'he_normal',strides=(1, 1))(pool2)
    pool1 = MaxPooling2D((2, 2), padding='same')(x1)
    x4 = Flatten()(pool1)
    # create LRG output

    output_shape=(batch_size,11,11,22)

    reshape1= Reshape(output_shape[1:])(x4)
    up1 = UpSampling2D((2, 2))(reshape1)
    l1=Conv2DTranspose(filters,kernel_size=num_conv,padding='same',strides=1,kernel_initializer = 'he_normal',activation='relu')(up1)
    up2 = UpSampling2D((2, 2))(l1)
    outputLRG=Conv2DTranspose(img_chns,kernel_size=2,padding='same',activation='sigmoid',name='outputLRG')(up2)

    # Source output

    reshape2= Reshape(output_shape[1:])(x4)
    up_1 = UpSampling2D((2, 2))(reshape2)
    s1=Conv2DTranspose(filters,kernel_size=num_conv,padding='same',strides=1,kernel_initializer = 'he_normal',activation='relu')(up_1)
    up_2 = UpSampling2D((2, 2))(s1)
    outputsource=Conv2DTranspose(img_chns,kernel_size=2,padding='same',activation='sigmoid',name='outputsource')(up_2)
    addlayer= Add()([outputLRG, outputsource])

    #final model
    model = Model(encoder_inputs, [outputLRG,outputsource,addlayer])
    return model,weights_name

def Two_steps_ae_with_classification(frozen_state=False):
    '''This model takes a lens candidate as input and outputs the foreground object image (LRG), the lensing features (source)
    and a classification.
    The model is aimed to be used in the second step of a two step training where the autoencoder part is trained first (ae_deblending_only model).
    The weights of the autoencoder parts can be kept frozen during training with the argument frozen_state

    '''
    weights_name='classifier_separate_trainingfreeze.h5'
    first_partmodel,weigtfirstpart=ae_deblending_only()
    first_partmodel.load_weights(weigtfirstpart)
    first_partmodel.trainable = frozen_state
    # classification output

    x30 = Flatten()(first_partmodel.get_layer('outputsource').output)
    dropout1 = Dropout(0.1)(x30)
    denseclass2 = Dense(100, activation='relu', kernel_initializer='he_normal')(dropout1)
    dropout = Dropout(0.1)(denseclass2)
    denseclass3 = Dense(1, activation='sigmoid')(dropout)
    modeltotal = Model(first_partmodel.get_layer('encoderinput').input, [first_partmodel.get_layer('outputLRG').output, first_partmodel.get_layer('outputsource').output,denseclass3])
    return modeltotal, weights_name

if __name__ == "__main__":
    X, Y1,Y2,Yclass =extract_data('E:\\data_autocoder\\training_set_correct_rescaling\\lenses\\','E:\\data_autocoder\\training_set_correct_rescaling\\nonlenses\\','E:\\data_autocoder\\training_set_correct_rescaling\\sources\\',
                     'E:\\data_autocoder\\training_set_correct_rescaling\\LRG\\')

    X_train, X_test, Y_train1, Y_test1, Y_train2, Y_test2,Y_classtrain,Y_classtest = train_test_split(X, Y1,Y2,Yclass, test_size=0.1, random_state=42)

    X_train = X_train.reshape((X_train.shape[0], 44, 44, 1))
    Y_train1 = Y_train1.reshape((X_train.shape[0], 44, 44, 1))
    Y_train2 = Y_train2.reshape((X_train.shape[0], 44, 44, 1))

    X_test= X_test.reshape((X_test.shape[0], 44, 44, 1))

    Y_test1= Y_test1.reshape((X_test.shape[0], 44, 44, 1))
    Y_test2= Y_test2.reshape((X_test.shape[0], 44, 44, 1))

    model,weights_name=addclassifieraftersource()#addclassifieraftersource()
    datagen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True, zca_whitening=False, zca_epsilon=1e-03)

    datagen.fit(X_train)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, min_delta=0.0001)

    def generate_data_generator(generator, X, Y1, Y2,Yclass):
        genX = generator.flow(X, Yclass, batch_size=100, seed=7)
        genY1 = generator.flow(Y1,Yclass,batch_size=100, seed=7)
        genY2 = generator.flow(Y2,Yclass,batch_size=100, seed=7)
        while True:
                Xi, yclass = genX.next()
                Yi1,y2 = genY1.next()
                Yi2,y3 = genY2.next()
                yield Xi, [Yi1, Yi2,Xi, yclass]
    def generate_data_generatorencoder(generator, X, Y1, Y2,Yclass):
        genX = generator.flow(X, Yclass, batch_size=100, seed=7)
        genY1 = generator.flow(Y1,Yclass,batch_size=100, seed=7)
        genY2 = generator.flow(Y2,Yclass,batch_size=100, seed=7)
        while True:
                Xi, yclass = genX.next()
                Yi1,y2 = genY1.next()
                Yi2,y3 = genY2.next()
                yield Xi, [Yi1, Yi2,Xi]
    def generate_data_generatowithoutlens(generator, X, Y1, Y2,Yclass):
        genX = generator.flow(X, Yclass, batch_size=100, seed=7)
        genY1 = generator.flow(Y1,Yclass,batch_size=100, seed=7)
        genY2 = generator.flow(Y2,Yclass,batch_size=100, seed=7)
        while True:
                Xi, yclass = genX.next()
                Yi1,y2 = genY1.next()
                Yi2,y3 = genY2.next()
                yield Xi, [Yi1, Yi2,yclass]


    model.compile(optimizer='adam',loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], loss_weights=[2, 4,2])#[2, 4, 0.1,2])# 'binary_crossentropy')
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True, min_delta=0.0001)
    model.fit_generator(generate_data_generatowithoutlens(datagen, X_train, Y_train1, Y_train2,Y_classtrain), steps_per_epoch = 100,epochs=1000,validation_data=generate_data_generatowithoutlens(datagen, X_test, Y_test1, Y_test2,Y_classtest),validation_steps=50, callbacks = [callback])
    model.save_weights(weights_name)
