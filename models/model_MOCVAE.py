# -*- coding: utf-8 -*-
# +
import json

import tensorflow as tf

from tensorflow.keras.layers import Input, Reshape, UpSampling2D, MaxPooling2D, add
from tensorflow.keras.layers import SeparableConv2D, Conv2DTranspose, Dense, Conv2D, Lambda, concatenate
from tensorflow.keras.layers import ReLU, Dropout, BatchNormalization, Flatten, Activation
from tensorflow.keras.activations import tanh
from tensorflow.keras import Model


# -

class Mocae(object):
    def __init__(self, config=None):
        with open('/workspace/Guille/MOC-AE/MOC-AE_Code/config.json', 'r') as f:
            config = json.load(f)
        
        # Models
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.classifier = None
        self.mocae = None

        # Networks configuration
        self.filters_encoder = config['model']['filters_encoder']
        self.filters_decoder = config['model']['filters_decoder']

        self.classifier_perceptron = config['model']['classifier_perceptron']

        self.latent_dim = config['model']['latent_dim']
        
        self.n_classes = len(config['padchest']['label_list'])
        
        # Image dimensions
        self.img_height =  config[config['experiment']]['image']['img_height']
        self.img_width = config[config['experiment']]['image']['img_width']
        self.img_channels = config[config['experiment']]['image']['img_channels']
    
    def sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], self.latent_dim), mean=0., stddev=0.1)
        return z_mean + tf.math.exp(z_log_sigma) * epsilon
    
    def create_encoder(self, input_img):
        x = input_img
        
        # Downsize layers
        for i in range(len(self.filters_encoder)):
            if i != 0:
                x = MaxPooling2D(3, strides=2, padding="same")(x)
            x = create_res_block(x, self.filters_encoder[i], kernel_size=3)

        # Latent dimension      
        fl = Flatten()(x)
        
        x = Dense(self.latent_dim)(fl)
        x = BatchNormalization()(x)
        x = tanh(x)
        z_mean = Dropout(0.3)(x)
        
        y = Dense(self.latent_dim)(fl)
        y = BatchNormalization()(y)
        y = tanh(y)
        z_log_sigma = Dropout(0.3)(y)
        
        return z_mean, z_log_sigma
    
    def create_decoder(self, latent_space):
        # From latent space to image
        height = int(self.img_height/(pow(2, len(self.filters_encoder)-1)))
        width = int(self.img_width/(pow(2, len(self.filters_encoder)-1)))
        channels = int(self.filters_decoder[0])
        
        x = Dense(height*width*channels)(latent_space)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.3)(x)
        x = Reshape((height, width, channels))(x)

        # Upsampling layers
        for i in range(len(self.filters_decoder)):
            if i != 0:
                x = UpSampling2D(2)(x)
            x = create_res_block(x, self.filters_decoder[i], kernel_size=3)
        
        # Output image generation
        x = SeparableConv2D(self.img_channels, kernel_size=1, strides=1, padding="same",
                            activation='tanh', name='rec')(x)
        return x
    
    def create_classifier(self, z_mean):
        # Perceptron layers
        x = Flatten()(z_mean)
        
        for n_neurons in self.classifier_perceptron:
            x = Dense(n_neurons)(x)
            x = ReLU()(x)
            x = Dropout(0.3)(x)

        x = Dense(self.n_classes, activation='softmax', name='class')(x)
        return x
    
    def create_mocae(self):
        input_img = Input(shape=(self.img_height,
                                 self.img_width,
                                 self.img_channels))

        # Encoder generation
        z_mean, z_log_sigma = self.create_encoder(input_img)
        z = Lambda(self.sampling)([z_mean, z_log_sigma])
        
        self.encoder = Model(input_img, [z_mean, z_log_sigma, z])

        # Decoder generation
        z_dec = Input(shape=(self.latent_dim,))
        output_img = self.create_decoder(z_dec)
        
        self.decoder = Model(z_dec, output_img, name='rec')
        
        reconstruction = self.decoder(self.encoder(input_img)[2])

        # Classifier generation
        z_clf = Input(shape=(self.latent_dim,))
        classification = self.create_classifier(z_clf)
        
        self.classifier = Model(z_clf, classification, name='class')
        
            # Use z as the classifier input
        classification = self.classifier(self.encoder(input_img)[2])

        # Autoencoder generation
        self.autoencoder = Model(input_img, reconstruction)
        
        # MOCAE generation
        self.mocae = Model(input_img, [reconstruction, classification])
        
        # KLDivergence loss
        kl_loss = -tf.reduce_mean(z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma) + 1) / 2
        self.mocae.add_loss(kl_loss)


def create_res_block(input_layer, filters, kernel_size):
    x = input_layer
    for i in range(2):
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.3)(x)
        x = SeparableConv2D(filters, kernel_size=kernel_size, strides=1, padding="same")(x)

    # Match num of filters
    y = SeparableConv2D(filters, kernel_size=(1,1), strides=1, padding="same")(input_layer)

    x = add([x, y])
    return x
