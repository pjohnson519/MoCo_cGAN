from keras.layers import Flatten, Dense, Input, Reshape, merge, Concatenate
from keras.layers.convolutional import Conv3D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
import keras.backend as K
import numpy as np
#from keras.utils import plot_model
def GanDiscriminator(output_img_dim):
    
    stride = 2
    bn_mode = 2
    axis = -1
    
    input_layer = Input(shape=[192,160,8,1])
    
    num_filters_start = 32
    nb_conv = int(np.floor(np.log(output_img_dim[1]) / np.log(2)))
    filters_list = [num_filters_start * min(8, (2 ** i)) for i in range(nb_conv)]

    # CONV 1
    disc_out = Conv3D(filters=64, kernel_size=4, padding='same', strides=(stride,stride, stride), name='disc_conv_1')(input_layer)
    disc_out = LeakyReLU(alpha=0.2)(disc_out)

    # CONV 2 - CONV N
    # do the rest of the convs based on the sizes from the filters
    for i, filter_size in enumerate(filters_list[1:]):
        name = 'disc_conv_{}'.format(i+2)

        disc_out = Conv3D(filters=filter_size, kernel_size=4, padding='same', strides=(stride,stride, stride), name=name)(disc_out)
        disc_out = BatchNormalization(name=name + '_bn', axis=axis)(disc_out)
        disc_out = LeakyReLU(alpha=0.2)(disc_out)

    # ------------------------
    # BUILD PATCH GAN
    # this is where we evaluate the loss over each sublayer of the input
    # ------------------------
    gan_discriminator = generate_gan_loss(disc_out,input_layer)
                                                      
    return gan_discriminator


def generate_gan_loss(last_disc_conv_layer,  input_layer):

    x_flat = Flatten()(last_disc_conv_layer)
    x = Dense(2, activation='softmax', name="disc_dense")(x_flat)

    

    discriminator = Model(input=last_disc_conv_layer, output=[x], name='discriminator_nn')
    return discriminator


