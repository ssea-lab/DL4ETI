from keras.layers import Activation, Reshape, Lambda, dot, add
from keras.layers import Conv1D, Conv2D, Conv3D
from keras.layers import MaxPool1D,GlobalAveragePooling2D,Dense,multiply,Activation,concatenate
from keras import backend as K


def squeeze_excitation_layer(x, out_dim, ratio = 4, concate = True):
    '''
    SE module performs inter-channel weighting.
    '''
    squeeze = GlobalAveragePooling2D()(x)

    excitation = Dense(units=out_dim // ratio)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=out_dim)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1, 1, out_dim))(excitation)

    scale = multiply([x, excitation])

    if concate:
        scale = concatenate([scale, x],axis=3)
    return scale
