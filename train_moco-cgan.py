import numpy as np
import os

from keras.models import Input, Model
from keras.layers import Activation, Input, Dropout, merge, Concatenate, MaxPooling3D
from keras.layers.convolutional import  Conv3D, UpSampling2D, UpSampling3D, Conv3DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers
from keras.models import load_model
from keras.optimizers import Adam
from utils.data_generator import  data_generator
from networks.discriminator import GanDiscriminator
from networks.CGAN import DCGAN
from utils import logger
from keras.utils.training_utils import multi_gpu_model
import time
from keras.utils import generic_utils as keras_generic_utils
import scipy

#data augmentation functions   

def random_rotation(image_array,image_motion_array):
    # pick a random degree of rotation 
    random_degree = np.random.uniform(-6, 6)
    return scipy.ndimage.interpolation.rotate(image_array, random_degree,  reshape=False), scipy.ndimage.interpolation.rotate(image_motion_array, random_degree, reshape=False)

# define the 3d unet architecture 

def conv_block(m, dim, acti, bn, res, do=0):
    n = Conv3D(dim, 3, activation=acti, padding='same')(m)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do)(n) if do else n
    n = Conv3D(dim, 3, activation=acti, padding='same')(n)
    n = BatchNormalization()(n) if bn else n
    return Concatenate()([m, n]) if res else n

def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
    if depth > 0:
        n = conv_block(m, dim, acti, bn, res,do=0)
        m = MaxPooling3D(pool_size=(2,2,2))(n) if mp else Conv3D(dim, 3, strides=2, padding='same')(n)
        m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
        
        if up:
                m = UpSampling3D(size=(2,2,2))(m)
                m = Conv3D(dim, 2, activation=acti, padding='same')(m)
        else:
                m = Conv3DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
        n = Concatenate()([n, m])
        m = conv_block(n, dim, acti, bn, res,do=0)
    else:
            m = conv_block(m, dim, acti, bn, res, do=0.5)
    return m
        
def UNet(img_shape, out_ch=1, start_ch=4, depth=2, inc_rate=2., activation='relu', dropout=0.5, batchnorm=False, maxpool=True, upconv=False, residual=False):
    i = Input(shape=img_shape)
    o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
    o = Conv3D(out_ch, 1, activation='relu')(o)
    return Model(inputs=i, outputs=o)

#load and prep data
images=np.load('/home/user/moco_dataset/targets.npy')
trainy=np.abs(np.expand_dims(images,4)) #add a channel dimension 

images_motion=np.load('/home/user/moco_dataset/inputs.npy')
trainx=np.abs(np.expand_dims(images_motion,4))


img_shape= [192,160,8,1] #(xdim,ydim,slices,channels)

#data augmentation options
fliplr=False
rotate=True

size = np.shape(trainx,0)
#split off 7 subjects for validation (have other subjects already reserved for testing)

valx=trainx[0:280,:,:,:,:] # 8 slabs * 5 motions * 7 subjects = 280
valy=trainy[0:280,:,:,:,:]
trainx=trainx[280:size,:,:,:,:]
trainy=trainy[280:size,:,:,:,:]


if fliplr:
    trainx=np.concatenate([trainx, np.flip(trainx,3)], axis=0)
    trainy=np.concatenate([trainy, np.flip(trainy,3)], axis=0)
    im_count=np.shape(trainx)[0]
    
#random shuffle of training examples
arr=np.arange(np.shape(trainx)[0])
np.random.shuffle(arr)
trainx=trainx[arr,...]
trainy=trainy[arr,...]

print('start augmentation rotations')
if rotate:
    im_count=np.shape(trainx)[0]
    tempx=np.zeros(np.shape(trainx))
    tempy=np.zeros(np.shape(trainy))
    for i in range(im_count):
        [tempx[i,:,:,:,:], tempy[i,:,:,:,:]]=random_rotation(trainx[i,:,:,:,:],trainy[i,:,:,:,:])
    trainx=np.concatenate([trainx,tempx])
    trainy=np.concatenate([trainy,tempy])

print('done data augmentation')
# width, height of images to work with.
im_width =192
im_height=160
im_slices=8

# input/oputputt channels in image
input_channels = 1
output_channels = 1

# image dims
input_img_dim =(im_width, im_height, im_slices, input_channels)
output_img_dim = ( im_width,im_height,im_slices,  output_channels)

# ----------------------
# GENERATOR
# Our generator is a 3D  U-NET with skip connections
# ----------------------

generator_nn =UNet(img_shape, out_ch=1, start_ch=64, depth=3, inc_rate=2., activation='relu', dropout=0.5, batchnorm=True, maxpool=True, upconv=True, residual=True)

generator_nn.summary()

# ----------------------
#  GAN DISCRIMINATOR
discriminator_nn = GanDiscriminator(output_img_dim=output_img_dim)
discriminator_nn.summary()

# disable training while we put it through the GAN
discriminator_nn.trainable = False

# ------------------------
# Define Optimizers
opt_discriminator = Adam(lr=5e-5,beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
opt_generator = Adam(lr=5E-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)

# -------------------------
# compile generator
generator_nn=multi_gpu_model(generator_nn, gpus=2)
generator_nn.compile(loss='mse', optimizer=opt_generator)

# ----------------------
# MAKE FULL CGAN
# ----------------------
cgan_nn = CGAN(generator_model=generator_nn,
                  discriminator_model=discriminator_nn,input_img_dim)

cgan_nn.summary()

# ---------------------
# Compile CGAN
# we use a combination of mae and bin_crossentropy
loss = ['mae', 'binary_crossentropy']

loss_weights = [1, 1]
cgan_nn=multi_gpu_model(cgan_nn,gpus=2)
cgan_nn.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_generator)

# ---------------------
discriminator_nn=multi_gpu_model(discriminator_nn,gpus=2)
discriminator_nn.trainable =True
discriminator_nn.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

# ------------------------
# RUN TRAINING
batch_size =4
nb_epoch = 50
n_images_per_epoch =np.shape(trainx)[0]

print('start training')
for epoch in range(0, nb_epoch):

    print('Epoch {}'.format(epoch))
    batch_counter = 1
    start = time.time()
    progbar = keras_generic_utils.Progbar(n_images_per_epoch)
    

    # init the datasources again for each epoch
    tng_gen = data_generator(trainx, trainy,batch_size=batch_size)
    val_gen = data_generator(valx,valy, batch_size=batch_size)


    for mini_batch_i in range(0, n_images_per_epoch, batch_size):

        
        X_train_decoded_imgs, X_train_original_imgs = next(tng_gen)
        
        # generate a batch of data and feed to the discriminator
        # some images that come out of here are real and some are fake
        # X is image patches for each image in the batch
        # Y is a 1x2 vector for each image. (means fake or not)
        X_discriminator, y_discriminator = patch_utils.get_disc_batch(X_train_original_imgs,
                                                          X_train_decoded_imgs,
                                                          generator_nn,
                                                          batch_counter,
                                                          patch_dim=sub_patch_dim)
        
        # Update the discriminator
        
        disc_loss = discriminator_nn.train_on_batch(X_discriminator, y_discriminator)

        # create a batch to feed the generator
        X_gen_target, X_gen = next(patch_utils.gen_batch(X_train_original_imgs, X_train_decoded_imgs, batch_size))
        y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
        y_gen[:, 1] = 1

        # Freeze the discriminator
        discriminator_nn.trainable = False

        # trainining GAN
        # print('calculating GAN loss...')
        gen_loss = cgan_nn.train_on_batch(X_gen, [X_gen_target, y_gen])

        # Unfreeze the discriminator
        
        discriminator_nn.trainable = True


        # counts batches we've ran through for generating fake vs real images
        batch_counter += 1

        # print losses
        D_log_loss = disc_loss
        gen_total_loss = gen_loss[0].tolist()
        gen_mae = gen_loss[1].tolist()
        gen_log_loss = gen_loss[2].tolist()
        

        progbar.add(batch_size, values=[("Dis logloss", D_log_loss),
                                        ("Gen total", gen_total_loss),
                                        ("Gen L1 (mae)", gen_mae),
                                        ("Gen logloss", gen_log_loss)])

        
    
    print('Epoch %s/%s, Time: %s' % (epoch + 1, nb_epoch, time.time() - start))
    
    # ------------------------------
    # save weights on every 10th epoch
    if epoch % 10 == 0:
        generator_nn.save('models_best')

    val_loss=generator_nn.evaluate(valx,valy,batch_size)
    
    print('val_loss = %s' %val_loss)
    

