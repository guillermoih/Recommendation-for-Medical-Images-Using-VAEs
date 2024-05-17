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

def train(config, model, dataset):
    batch_size=config['train']['batch_size']
    n_epochs=config['train']['n_epochs']
    conf_mat_samples = config['train']['conf_mat_samples']
    
    min_val_mean_loss = 999
    
    history = {
        'loss' : [],
        'val_loss' : [],
        'loss_mean' : [],
        'loss_val_mean' : [],
        'loss_epoch' : [],

        'rec_loss' : [],
        'val_rec_loss' : [],
        'rec_loss_mean' : [],
        'rec_loss_val_mean' : [],
        'rec_loss_epoch' : [],

        'class_loss' : [],
        'val_class_loss' : [],
        'class_loss_mean' : [],
        'class_loss_val_mean' : [],
        'class_loss_epoch' : [],
    }
    
    print('-----COMPILING MODEL-----')
    losses = {
        "rec": "mse",
        "class": "categorical_crossentropy",
    }
        
    model.mocae.compile(loss=losses,
                        loss_weights=config["train"]["loss_weights"],
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
                y_labels = dataset.y_train[batch*batch_size:(batch+1)*batch_size]
                sample_weight = dataset.weight_train[batch*batch_size:(batch+1)*batch_size]

                losses = model.mocae.train_on_batch(x=X,
                                                    y={"rec": X, "class": y_labels},
                                                    return_dict=True,
                                                    sample_weight={"class":sample_weight})
                                                    
                history['loss'].append(losses['loss'])
                history['rec_loss'].append(losses['rec_loss'])
                history['class_loss'].append(losses['class_loss'])

                idx = np.random.randint(0, len(dataset.X_val) / batch_size)
                X_val = dataset.X_val[idx*batch_size:(idx+1)*batch_size]
                y_labels = dataset.y_val[idx*batch_size:(idx+1)*batch_size]
                sample_weight = dataset.weight_val[idx*batch_size:(idx+1)*batch_size]

                val_losses = model.mocae.test_on_batch(x=X_val,
                                                       y={"rec": X_val, "class": y_labels},
                                                       return_dict=True,
                                                       sample_weight={"class":sample_weight})
                    
                history['val_loss'].append(val_losses['loss'])
                history['val_rec_loss'].append(val_losses['rec_loss'])
                history['val_class_loss'].append(val_losses['class_loss'])

            # Reconstruction evaluation
            n_plots = 3
            plot_reconstruction(n_plots, dataset.X_val[:n_plots],
                                model.autoencoder.predict(dataset.X_val[:n_plots]), batch, epoch,
                                config["PATHS"]["LOG_PATH"])

            # Confusion matrix evaluation
            y_real = dataset.y_val[:conf_mat_samples]
            lat_space = model.encoder.predict(dataset.X_val[:conf_mat_samples])
            y_pred = model.classifier.predict(lat_space)
            #classes_conf_matrix(y_real, np.around(y_pred, 0), batch, epoch)

            y_real = np.argmax(y_real, axis=1)
            y_pred = np.argmax(y_pred, axis=1)

            labels = sorted(config["padchest"]["label_names"], key=config["padchest"]["label_names"].get)
            conf_matrix(y_real, np.around(y_pred, 0), labels, batch, epoch, config["PATHS"]["LOG_PATH"])

            get_mean_loss('loss', n_batches, history)
            get_mean_loss('rec_loss', n_batches, history)
            get_mean_loss('class_loss', n_batches, history)

            plot_train_val(history, LOG_PATH=config["PATHS"]["LOG_PATH"])
            plot_train_val(history, 'rec_', LOG_PATH=config["PATHS"]["LOG_PATH"])
            plot_train_val(history, 'class_', LOG_PATH=config["PATHS"]["LOG_PATH"])
            
            # Save best model and conf matrix (validation total loss)
            if history["loss_val_mean"][-1] < min_val_mean_loss:
                min_val_mean_loss = history["loss_val_mean"][-1]
                
                model.encoder.save(config["PATHS"]["LOG_PATH"] + 'models/e_best_encoder.h5')
                model.decoder.save(config["PATHS"]["LOG_PATH"] + 'models/e_best_decoder.h5')
                model.autoencoder.save(config["PATHS"]["LOG_PATH"]  + 'models/e_best_autoencoder.h5')
                model.classifier.save(config["PATHS"]["LOG_PATH"]  + 'models/e_best_classifier.h5')
                model.mocae.save(config["PATHS"]["LOG_PATH"]  + 'models/e_best_mocae.h5')
                
                y_real = dataset.y_val[:conf_mat_samples]
                lat_space = model.encoder.predict(dataset.X_val[:conf_mat_samples])
                y_pred = model.classifier.predict(lat_space)

                y_real = np.argmax(y_real, axis=1)
                y_pred = np.argmax(y_pred, axis=1)

                labels = sorted(config["padchest"]["label_names"], key=config["padchest"]["label_names"].get)
                conf_matrix(y_real, np.around(y_pred, 0), labels, batch, epoch,
                            config["PATHS"]["LOG_PATH"], best=True)

            # Saving models
            if epoch%10 == 0:
                model.encoder.save(config["PATHS"]["LOG_PATH"] + 'models/e' + str(epoch).zfill(3) + '_encoder.h5')
                model.decoder.save(config["PATHS"]["LOG_PATH"] + 'models/e' + str(epoch).zfill(3) + '_decoder.h5')
                model.autoencoder.save(config["PATHS"]["LOG_PATH"]  + 'models/e' + str(epoch).zfill(3) + '_autoencoder.h5')
                model.classifier.save(config["PATHS"]["LOG_PATH"]  + 'models/e' + str(epoch).zfill(3) + '_classifier.h5')
                model.mocae.save(config["PATHS"]["LOG_PATH"]  + 'models/e' + str(epoch).zfill(3) + '_mocae.h5')


# +
# WARNING, used in two files (dataset and train)

def shuffle_three_arrays(a, b, c):
    '''assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]'''
    
    combined = list(zip(a, b, c))
    random.shuffle(combined)
    a_permuted, b_permuted, c_permuted = zip(*combined)
    
    return np.array(a_permuted), np.array(b_permuted), np.array(c_permuted)
