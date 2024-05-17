# -*- coding: utf-8 -*-
# +
import os, glob, json
import pandas as pd
import numpy as np

from PIL import Image

# +
with open('/workspace/Guille/MOC-AE/MOC-AE_Code/config.json', 'r') as f:
    config = json.load(f)

IMAGES_PATH = config['PATHS']['IMAGES_PATH']


# -

class Dataset(object):
    def __init__(self, config):
        self.X_train = None
        self.X_seg_train = None
        self.y_train = None
        self.y_train_imgs = None
        
        self.X_val = None
        self.X_seg_val = None
        self.y_val = None
        self.y_val_imgs = None
        
        self.sub_id_train = None
        self.tumour_area_train = None
        self.sub_id_val = None
        self.tumour_area_val = None
        
        self.val_perc = config['train']['val_perc']
        
    def load_data(self):
        X_load = []
        X_seg_load = []
        y_load = []
        y_load_imgs = []
        sub_id_load = []
        tumour_area = []

        for i, folder_path in enumerate(glob.glob(IMAGES_PATH + '/*')):
            if i%75==0:
                print(str(i) + ' folders loaded')

            id = folder_path[-3:]

            for img_path in glob.glob(folder_path + '/*flair*.png'):
                img = Image.open(img_path)
                img_array = np.array(img)
                img_array = np.expand_dims(img_array, axis=0)
                
                img_array = np.append(img_array, [np.array(Image.open(img_path[:81] + 't1' + img_path[-8:]))], axis=0)
                img_array = np.append(img_array, [np.array(Image.open(img_path[:81] + 't1ce' + img_path[-8:]))], axis=0)
                img_array = np.append(img_array, [np.array(Image.open(img_path[:81] + 't2' + img_path[-8:]))], axis=0)
                
                img_array = np.rollaxis(img_array, 0, 3)
                X_load.append(img_array)
                
                path = img_path[:81] + 'hemi' + img_path[-8:]
                if os.path.exists(path):
                    img = Image.open(path)
                    img_array = np.array(img)
                    X_seg_load.append(img_array)
                else:
                    X_seg_load.append(None)

                y_path = img_path[:81] + 'seg' + img_path[-8:]
                img = Image.open(y_path)
                img_array = np.array(img)
                y_load_imgs.append(img_array)

                sub_id_load.append(id)
                tumour_area.append(get_tum_area(img_array))
                
                if get_tum_area(img_array)>0:
                    y_load.append(1)
                    
                else:
                    y_load.append(0)

        # Input image scanner
        X_train_aux = np.asarray(X_load[int(self.val_perc * len(X_load)):])
        X_train_aux = X_train_aux/127.5 - 1
        self.X_train = X_train_aux
        
        X_val_aux = np.asarray(X_load[:int(self.val_perc * len(X_load))])
        X_val_aux = X_val_aux/127.5 - 1
        self.X_val = X_val_aux
        
        # Segmented anatomical labels
        X_seg_train_aux = np.asarray(X_seg_load[int(self.val_perc * len(X_seg_load)):])
        self.X_seg_train = X_seg_train_aux
        
        X_seg_val_aux = np.asarray(X_seg_load[:int(self.val_perc * len(X_seg_load))])
        self.X_seg_val = X_seg_val_aux
        
        # Tumour presence labels
        y_train_aux = np.asarray(y_load[int(self.val_perc * len(y_load_imgs)):])
        self.y_train = y_train_aux
        
        y_val_aux = np.asarray(y_load[:int(self.val_perc * len(y_load))])
        self.y_val = y_val_aux
        
        # Tumour segmented images
        y_train_imgs_aux = np.asarray(y_load_imgs[int(self.val_perc * len(y_load_imgs)):])
        y_train_imgs_aux = y_train_imgs_aux/64
        self.y_train_imgs = y_train_imgs_aux
        
        y_val_imgs_aux = np.asarray(y_load_imgs[:int(self.val_perc * len(y_load_imgs))])
        y_val_imgs_aux = y_val_imgs_aux/64
        self.y_val_imgs = y_val_imgs_aux
        
        # Patient id
        self.sub_id_train = np.asarray(sub_id_load[int(self.val_perc * len(sub_id_load)):])
        self.sub_id_val = np.asarray(sub_id_load[:int(self.val_perc * len(sub_id_load))])
        
        # Tumoural area
        self.tumour_area_train = np.asarray(tumour_area[int(self.val_perc * len(tumour_area)):])
        self.tumour_area_val = np.asarray(tumour_area[:int(self.val_perc * len(tumour_area))])


def load_labels(y):
    labels = []
    for img in y:
        if np.amax(img) > 0:
            labels.append(1)
        else:
            labels.append(0)

    return np.asarray(labels)


def get_tum_area(img_array):
    return np.count_nonzero(img_array > 0)
