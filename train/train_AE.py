# -*- coding: utf-8 -*-
# +
import sys
sys.path.append("..")

import json, random

from utils.metrics import *

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow import device

from dataset.dataset_padchest import shuffle_three_arrays


# -

def train(config, model, dataset, mocae=False):
    batch_size=config['train']['batch_size']
    n_epochs=config['train']['n_epochs']
    
    min_val_mean_loss = 999
    
    history = {
        'loss' : [],
        'val_loss' : [],
        'loss_mean' : [],
        'loss_val_mean' : [],
        'loss_epoch' : [],
    }
    
    print('-----COMPILING MODEL-----')
    model.autoencoder.compile(loss="mse",
                        optimizer=Adam(learning_rate=config['train']['learning_rate']))
                    
    print('-----TRAIN START-----')
    with device('/GPU:0'):
        for epoch in range(n_epochs):
            dataset.epoch = epoch
            
            dataset.X_train, dataset.y_train, dataset.weight_train = shuffle_three_arrays(dataset.X_train, dataset.y_train, dataset.weight_train)
            dataset.X_val, dataset.y_val, dataset.weight_val = shuffle_three_arrays(dataset.X_val, dataset.y_val, dataset.weight_val)

            n_batches = int(len(dataset.X_train) / batch_size)
            for batch in range(n_batches):
                X = dataset.X_train[batch*batch_size:(batch+1)*batch_size]

                losses = model.autoencoder.train_on_batch(x=X, y=X, return_dict=True)
                                                    
                history['loss'].append(losses['loss'])

                idx = np.random.randint(0, len(dataset.X_val) / batch_size)
                X_val = dataset.X_val[idx*batch_size:(idx+1)*batch_size]

                val_losses = model.autoencoder.test_on_batch(x=X_val, y=X_val, return_dict=True)
                    
                history['val_loss'].append(val_losses['loss'])
                
            # Reconstruction evaluation
            n_plots = 3
            plot_reconstruction(n_plots, dataset.X_val[:n_plots],
                                model.autoencoder.predict(dataset.X_val[:n_plots]), batch, epoch,
                                config["PATHS"]["LOG_PATH"])

            get_mean_loss('loss', n_batches, history)

            plot_train_val(history, LOG_PATH=config["PATHS"]["LOG_PATH"])
            
            # Save best model (validation total loss)
            if history["loss_val_mean"][-1] < min_val_mean_loss:
                min_val_mean_loss = history["loss_val_mean"][-1]

                model.encoder.save(config["PATHS"]["LOG_PATH"] + 'models/e_best_encoder.h5')
                model.autoencoder.save(config["PATHS"]["LOG_PATH"]  + 'models/e_best_autoencoder.h5')
                
                if mocae:
                    model.classifier.save(config["PATHS"]["LOG_PATH"]  + 'models/e_best_classifier.h5')
                    model.mocae.save(config["PATHS"]["LOG_PATH"]  + 'models/e_best_mocae.h5')

            # Saving models
            if epoch%10 == 0:
                model.encoder.save(config["PATHS"]["LOG_PATH"] + 'models/e' + str(epoch).zfill(3) + '_encoder.h5')
                model.autoencoder.save(config["PATHS"]["LOG_PATH"]  + 'models/e' + str(epoch).zfill(3) + '_autoencoder.h5')
                
                if mocae:
                    model.classifier.save(config["PATHS"]["LOG_PATH"]  + 'models/e' + str(epoch).zfill(3) + '_classifier.h5')
                    model.mocae.save(config["PATHS"]["LOG_PATH"]  + 'models/e' + str(epoch).zfill(3) + '_mocae.h5')
