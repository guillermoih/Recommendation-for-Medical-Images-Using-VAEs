# -*- coding: utf-8 -*-
# +
import os, glob, time, json, itertools, random
from datetime import datetime
import pandas as pd
import numpy as np

from PIL import Image, ImageFile
from sklearn.preprocessing import MultiLabelBinarizer
# -

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.IMAGES_PATH = self.config['PATHS']['IMAGES_PATH']
        
        self.df = None
        
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        
        self.img_height =  self.config['padchest']['image']['img_height']
        self.img_width = self.config['padchest']['image']['img_width']
        self.img_channels = self.config['padchest']['image']['img_channels']
        
        # Dataset labels
        self.label_list = self.config['padchest']['label_list']
        
        self.val_perc = self.config['train']['val_perc']
        
    def load_padchest_df(self, verbose=0):
        print('-----GENERATING DATAFRAME-----')
        # Load data from .csv
        self.df = pd.read_csv(self.IMAGES_PATH + 'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv',
                        converters={'WindowCenter_DICOM': convert_dtype, 'WindowWidth_DICOM': convert_dtype})

        # Drop images fron non downloaded folders
        downloaded_folders = list(range(0, self.config['padchest']['folder_num']))
        self.df = self.df.drop(self.df[~self.df.ImageDir.isin(downloaded_folders)].index)

        # Drop images with strange positions
        projection = ['PA']
        self.df = self.df.drop(self.df[~self.df.Projection.isin(projection)].index)
        
        # Drop images automatically labelled
        #method_label = ['Physician']
        #self.df = self.df.drop(self.df[~self.df.MethodLabel.isin(method_label)].index)

        # Drop images with strange labels
        self.df = self.df[self.df['Labels'].notna()]
        index_drop = []
        
        self.df['Labels'] = self.df['Labels'].apply(self.filter_values)
        self.df = self.df[self.df['Labels'].apply(len) == 1]
        self.df['Labels'] = self.df['Labels'].apply(lambda x: str(x))
        
        # Undersampling labels
        if self.config["train"]["undersampling"]:
            min_class = np.min(self.df['Labels'].value_counts().values)
            self.df = self.df.groupby('Labels').apply(lambda x: x.sample(n=min_class)).reset_index(drop=True)
            
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
            
        with open(self.config["json_path"], 'w') as f:
            json.dump(self.config, f, indent = 4)

        if verbose==1:
            print('---IMAGE FOLDERS---\n' + str(self.df['ImageDir'].value_counts(dropna=False)))
            print('\n---IMAGE PROJECTIONS---\n' + str(self.df['Projection'].value_counts(dropna=False)))
            print('\n---IMAGE LABELS---\n' + str(self.df['Labels'].value_counts(dropna=False)))
        
    def load_data(self, verbose=0):
        self.load_padchest_df(verbose=verbose)
        
        print('-----LOADING DATA FROM DF-----')
        # Get list of unused labels
        labels_to_be_removed = remove_elements(self.get_all_labels(), self.label_list)

        # Create list of image path and labels
        imgs_list = []
        label_list = []
        weight_list = []
        img_id_list = []
        
        # Additional fields
        age_list = []
        sex_list = []

        for i, row in enumerate(self.df.itertuples()):
            if i%1000 == 0:
                print(time.strftime("%H:%M:%S", time.localtime()))
                print(str(i) + ' images loaded')

            img = self.load_img_from_path(self.IMAGES_PATH + str(row.ImageDir) + '/' + row.ImageID)
            imgs_list.append(img)
            
            row_list = eval(row.Labels)
            label_list.append(row_list)
            
            img_id_list.append(str(row.ImageDir) + '/' + row.ImageID)
            
            # Get additional features
            study_date = datetime.strptime(str(row.StudyDate_DICOM), '%Y%m%d')
            birth = int(row.PatientBirth)
            age = study_date.year - birth
            age_list.append(age)
            sex_list.append(str(row.PatientSex_DICOM))
            
        # Save class weights
        total_samples = len(self.df['Labels'])
        dict_weights = {}
        for label, n_samples in self.df['Labels'].value_counts(dropna=False).items():
            dict_weights[eval(label)[0]] = total_samples/(n_samples*2)
            
        weight_list = [dict_weights[elem[0]] for elem in label_list]
        
        # Binarize label list
        binarizer = MultiLabelBinarizer(classes=list(self.config["padchest"]["label_list"]))
        label_list = binarizer.fit_transform(label_list)

        labels = [[x] for x in self.config["padchest"]["label_list"]]
        label_vectors = binarizer.fit_transform(labels)
        
        for label, value in zip(self.config["padchest"]["label_list"], np.argmax(label_vectors, axis=1)):
            self.config["padchest"]["label_names"][label] = value
        
        with open(self.config["json_path"], 'w') as f:
            json.dump(self.config, f, indent = 4, default=int)
        
        # Create data partitions
        
        #imgs_list, label_list, weight_list, img_id_list = shuffle_three_arrays(imgs_list, label_list, weight_list, img_id_list)
        
        self.X_train = np.array(imgs_list[int(self.val_perc * len(imgs_list)):])
        self.X_train = self.X_train/ (self.config["padchest"]["pixel_depth"]/2) - 1
        self.X_train = np.expand_dims(self.X_train, axis=-1)
        self.X_val = np.array(imgs_list[:int(self.val_perc * len(imgs_list))])
        self.X_val = self.X_val/(self.config["padchest"]["pixel_depth"]/2) - 1
        self.X_val = np.expand_dims(self.X_val, axis=-1)

        self.y_train = label_list[int(self.val_perc * len(label_list)):]
        self.y_train = np.asarray(self.y_train)
        self.y_val = label_list[:int(self.val_perc * len(label_list))]
        self.y_val = np.asarray(self.y_val)
        
        self.weight_train = weight_list[int(self.val_perc * len(weight_list)):]
        self.weight_val = weight_list[:int(self.val_perc * len(weight_list))]
        
        self.id_train = img_id_list[int(self.val_perc * len(img_id_list)):]
        self.id_val = img_id_list[:int(self.val_perc * len(img_id_list))]
        
        self.age_train = age_list[int(self.val_perc * len(age_list)):]
        self.age_val = age_list[:int(self.val_perc * len(age_list))]
        
        self.sex_train = sex_list[int(self.val_perc * len(sex_list)):]
        self.sex_val = sex_list[:int(self.val_perc * len(sex_list))]

        if verbose==1:
            print("---DATA PARTITION---\n")
            print("X_train: " + str(np.shape(self.X_train)))
            print("y_train: " + str(np.shape(self.y_train)))
            print("X_val: " + str(np.shape(self.X_val)))
            print("y_val: " + str(np.shape(self.y_val)))
            
    def get_all_labels(self):
        labels = []
        for index, row in self.df.iterrows():
            labels.append(eval(row['Labels']))

        flattened_labels = list(itertools.chain(*labels))
        return set(flattened_labels)
    
    def load_img_from_path(self, path):
        img = Image.open(path)
        img = img.resize((self.img_width, self.img_height), Image.BILINEAR)
        img_array = np.array(img)
        img.close()

        return img_array
    
    def filter_values(self, row):
        unique_values = set(eval(row))

        valid_values = unique_values.intersection(set(self.label_list))
        new_row = np.array(list(valid_values))

        return new_row


def remove_elements(input_list, elem_list):
    for elem in elem_list:
        if elem in input_list:
            input_list.remove(elem)
        
    return input_list


def convert_dtype(x):
    if str(x)=='None':
        return 0
    else:
        return x


def flatten(array_values):
    flattened = []
    for i in array_values:
        if isinstance(i,list): flattened.extend(flatten(i))
        else: flattened.append(i)
    return flattened


def shuffle_three_arrays(a, b, c):
    '''assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]'''
    
    combined = list(zip(a, b, c))
    random.shuffle(combined)
    a_permuted, b_permuted, c_permuted = zip(*combined)
    
    return np.array(a_permuted), np.array(b_permuted), np.array(c_permuted)
