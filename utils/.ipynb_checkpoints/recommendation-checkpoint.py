# -*- coding: utf-8 -*-
# +
import json

import cv2

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import euclidean
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MultiLabelBinarizer
from skimage import exposure

from statistics import mean, stdev
# -

with open('/workspace/Guille/MOC-AE/MOC-AE_Code/config.json', 'r') as f:
    config = json.load(f)


def get_nn(dataset, classifier, latent_representations, img_index):
    source_img = latent_representations[img_index]
    source_id = dataset.sub_id_train[img_index]

    X_neighbors = []
    X_seg_neighbors = []
    y_neighbors = []
    id_neighbors = []
    distances = []

    tum_pred = classifier.predict(np.expand_dims(dataset.X_train[img_index], axis=0))

    for i, candidate in enumerate(latent_representations):
        if dataset.sub_id_train[i] != source_id and dataset.sub_id_train[i] not in id_neighbors and dataset.tumour_area_train[i] != 0:
            if tum_pred>0.90:
                if dataset.tumour_area_train[i] != 0:
                    distances.append(euclidean(source_img, candidate))

                    X_neighbors.append(dataset.X_train[i])
                    X_seg_neighbors.append(dataset.X_seg_train[i])
                    y_neighbors.append(dataset.y_train_imgs[i])
                    id_neighbors.append(dataset.sub_id_train[i])
            else:
                distances.append(euclidean(source_img, candidate))

                X_neighbors.append(dataset.X_train[i])
                X_seg_neighbors.append(dataset.X_seg_train[i])
                y_neighbors.append(dataset.y_train_imgs[i])
                id_neighbors.append(dataset.sub_id_train[i])

    return distances, np.asarray(X_neighbors), np.asarray(X_seg_neighbors), np.asarray(y_neighbors), np.asarray(id_neighbors)


def plot_nearest_neighbor(neighbors, source_X, source_X_seg, source_y, source_id, num_neighbors):
    f, ax = plt.subplots(3, num_neighbors + 1, figsize=(8*num_neighbors, 18))

    ax[0][0].set_title('Query image')
    ax[0][0].imshow(source_X[:,:,3], cmap='gray', vmin=0, vmax=1)
    ax[0][0].get_xaxis().set_visible(False)
    ax[0][0].get_yaxis().set_visible(False)

    if np.any(source_X_seg) != None:
        ax[1][0].set_title('Query image')
        ax[1][0].imshow(source_X_seg, cmap='magma', vmin=0, vmax=256)
        ax[1][0].get_xaxis().set_visible(False)
        ax[1][0].get_yaxis().set_visible(False)

    ax[2][0].set_title('Query image')
    ax[2][0].imshow(source_y, cmap='magma', vmin=0, vmax=4)
    ax[2][0].get_xaxis().set_visible(False)
    ax[2][0].get_yaxis().set_visible(False)

    for i in range(num_neighbors):
        ax[0][i+1].imshow(neighbors[i][1][:,:,3], cmap='gray', vmin=0, vmax=1)
        ax[0][i+1].get_xaxis().set_visible(False)
        ax[0][i+1].get_yaxis().set_visible(False)

        if np.any(neighbors[i][2]) != None:
            ax[1][i+1].imshow(neighbors[i][2], cmap='magma', vmin=0, vmax=256)
            ax[1][i+1].get_xaxis().set_visible(False)
            ax[1][i+1].get_yaxis().set_visible(False)

        ax[2][i+1].imshow(neighbors[i][3], cmap='magma', vmin=0, vmax=4)
        ax[2][i+1].get_xaxis().set_visible(False)
        ax[2][i+1].get_yaxis().set_visible(False)

    plt.show()


def recom_system(dataset, classifier, X_low, img_idx, n_neighbors):
    distances, X_neighbors, X_seg_neighbors, y_neighbors, id_neighbors = get_nn(dataset, classifier, X_low, img_index=img_idx)
    neighbors = zip(distances, X_neighbors, X_seg_neighbors, y_neighbors, id_neighbors)
    neighbors = sorted(neighbors, key=lambda x: x[0])
    neighbors = list(neighbors)

    plot_nearest_neighbor(neighbors,
                               dataset.X_train[img_idx], dataset.X_seg_train[img_idx], dataset.y_train_imgs[img_idx], dataset.sub_id_train[img_idx],
                               num_neighbors=n_neighbors)


def get_nn_padchest(latent_representations, img_index):
    source_z = latent_representations[img_index]

    indexes = []
    distances = []

    for i, candidate in enumerate(latent_representations):
        if i != img_index:
            distances.append(euclidean(source_z, candidate))
            indexes.append(i)

    neighbors = zip(distances, indexes)
    neighbors = sorted(neighbors, key=lambda x: x[0])
    neighbors = list(neighbors)
    
    return neighbors


# Recover the nearest neighbor list filtering all labels except "label_base" and normal
def get_nn_padchest_filter_normal(data, labels, latent_representations, label_base, img_index):
    source_z = latent_representations[img_index]

    indexes = []
    distances = []

    for i, candidate in enumerate(latent_representations):
        #if labels[np.argmax(data.y_val[i])] == label_base or labels[np.argmax(data.y_val[i])] == 'normal':
        if labels[np.argmax(data.y_val[i])] != 'normal':
            if i != img_index:
                distances.append(euclidean(source_z, candidate))
                indexes.append(i)

    neighbors = zip(distances, indexes)
    neighbors = sorted(neighbors, key=lambda x: x[0])
    neighbors = list(neighbors)
    
    return neighbors


def plot_nn_padchest(ax, neighbors, data, img_idx, num_neighbors,
                     show_age=False, show_sex=False):
    ax[0].set_title('ID: '+ str(img_idx) + '\nQuery image')
    ax[0].imshow(data.X_train[img_idx,:,:], cmap='gray', vmin=-1, vmax=1)
    ax[0].get_yaxis().set_visible(False)
    ax[0].set_xticks([])
    
    labels = sorted(config["padchest"]["label_names"], key=config["padchest"]["label_names"].get)
    x_label_str = labels[np.argmax(data.y_train[img_idx])]
    if show_age != False:
        x_label_str = x_label_str + '\nAge: ' + str(data.age_train[img_idx])
    if show_sex != False:
        x_label_str = x_label_str + '\nSex: ' + data.sex_train[img_idx]
    ax[0].set_xlabel(x_label_str)

    for i in range(num_neighbors):
        ax[i+1].imshow(data.X_train[neighbors[i][1],:,:], cmap='gray', vmin=-1, vmax=1)
        ax[i+1].get_yaxis().set_visible(False)
        ax[i+1].set_xticks([])
        
        ax[i+1].set_title('ID: '+ str(neighbors[i][1]) +
                          '\nDist: '+ str(wass_distance(data.X_train[img_idx,:,:], data.X_train[neighbors[i][1],:,:])))
        
        x_label_str = labels[np.argmax(data.y_train[neighbors[i][1]])]
        if show_age != False:
            x_label_str = x_label_str + '\nAge: ' + str(data.age_train[neighbors[i][1]])
        if show_sex != False:
            x_label_str = x_label_str + '\nSex: ' + data.sex_train[neighbors[i][1]]
            
        ax[i+1].set_xlabel(x_label_str)


# BORRAR, solo es util para generar imagenes con los resultados
def plot_nn_padchest_save(ax, neighbors, data, img_idx, num_neighbors,
                          show_age=False, show_sex=False):
    ax[0].set_title('ID: '+ str(img_idx) + '\nQuery image')

    img = plt.imread(config["PATHS"]["HIGH_RES_IMAGES_PATH"] + data.id_train[img_idx])
    ax[0].imshow(img, cmap='gray')
    ax[0].get_yaxis().set_visible(False)
    ax[0].set_xticks([])
    
    labels = sorted(config["padchest"]["label_names"], key=config["padchest"]["label_names"].get)
    
    x_label_str = labels[np.argmax(data.y_train[img_idx])]
    if show_age != False:
        x_label_str = x_label_str + '\nAge: ' + str(data.age_train[img_idx])
    if show_sex != False:
        x_label_str = x_label_str + '\nSex: ' + data.sex_train[img_idx]
    ax[0].set_xlabel(x_label_str)

    for i in range(num_neighbors):
        img = plt.imread(config["PATHS"]["HIGH_RES_IMAGES_PATH"] + data.id_train[neighbors[i][1]])
        ax[i+1].imshow(img, cmap='gray')
        ax[i+1].get_yaxis().set_visible(False)
        ax[i+1].set_xticks([])
        
        ax[i+1].set_title('ID: '+ str(neighbors[i][1]) +
                          '\nDist: '+ str(wass_distance(data.X_train[img_idx,:,:], data.X_train[neighbors[i][1],:,:])))
        
        x_label_str = labels[np.argmax(data.y_train[neighbors[i][1]])]
        if show_age != 0:
            x_label_str = x_label_str + '\nAge: ' + str(data.age_train[neighbors[i][1]])
        if show_sex != 0:
            x_label_str = x_label_str + '\nSex: ' + data.sex_train[neighbors[i][1]]
            
        ax[i+1].set_xlabel(x_label_str)


def plot_nneighbors(data, model, n_neighbors=5, n_cases=5, vae=False):
    lat_rep = model.predict(data.X_train)
    
    f, ax = plt.subplots(n_cases, n_neighbors + 1, figsize=(5*n_neighbors, 5*n_cases))

    for i in range(n_cases):
        # Use the z mean for VAE models, in other case use all z
        if vae:
            neighbor_list = get_nn_padchest(lat_rep[0], img_index=i)
        else:
            neighbor_list = get_nn_padchest(lat_rep, img_index=i)
            
        plot_nn_padchest(ax[i], neighbor_list, data, i, n_neighbors)
        
    plt.show()


def plot_nneighbors_filtered(path, data, model, n_neighbors=5, n_cases=5, vae=False, label="normal"):
    cont = 0
    i = 0
    
    lat_rep = model.predict(data.X_train)
    
    f, ax = plt.subplots(n_cases, n_neighbors + 1, figsize=(9*n_neighbors, 9*n_cases), dpi=100)
        
    labels = sorted(config["padchest"]["label_names"], key=config["padchest"]["label_names"].get)
    
    while cont<n_cases and i<len(data.X_train):
        
        img_label = labels[np.argmax(data.y_train[i])]

        if(img_label == label):
            # Use the z mean for VAE models, in other case use all z
            if vae:
                neighbor_list = get_nn_padchest(lat_rep[0], img_index=i)
            else:
                neighbor_list = get_nn_padchest(lat_rep, img_index=i)
                
            plot_nn_padchest(ax[cont], neighbor_list, data, i, n_neighbors)
            
            cont = cont+1

        i = i+1
    
    plt.savefig(path)
    plt.show()


# BORRAR, solo es util para generar imagenes con los resultados
def plot_nneighbors_filtered_original(path, data, model, n_neighbors=5, n_cases=5, vae=False, label="normal"):
    cont = 0
    i = 0
    
    lat_rep = model.predict(data.X_train)
    
    f, ax = plt.subplots(n_cases, n_neighbors + 1, figsize=(15*n_neighbors, 15*n_cases), dpi=100)
    plt.rc('font', size=25)
        
    labels = sorted(config["padchest"]["label_names"], key=config["padchest"]["label_names"].get)
    
    while cont<n_cases and i<len(data.X_train):
        
        img_label = labels[np.argmax(data.y_train[i])]

        if(img_label == label):
            # Use the z mean for VAE models, in other case use all z
            if vae:
                neighbor_list = get_nn_padchest(lat_rep[0], img_index=i)
            else:
                neighbor_list = get_nn_padchest(lat_rep, img_index=i)
                
            plot_nn_padchest_save(ax[cont], neighbor_list, data, i, n_neighbors)
            
            cont = cont+1

        i = i+1
    
    plt.savefig(path)
    plt.show()


def plot_nneighbors_filtered_stratify(path, data, model, n_neighbors=5, n_cases=5, vae=False, label="normal",
                                      age=0, sex=0):
    cont = 0
    i = 0
    enc=0
    
    lat_rep = model.predict(data.X_train)
    
    f, ax = plt.subplots(n_cases, n_neighbors + 1, figsize=(9*n_neighbors, 9*n_cases), dpi=100)
        
    labels = sorted(config["padchest"]["label_names"], key=config["padchest"]["label_names"].get)
    
    while cont<n_cases and i<len(data.X_train):
        if age==1 and data.age_train[i] >= 65:
            enc=1
        elif age==2 and data.age_train[i] < 65:
            enc=1
        elif sex==1 and data.sex_train[i] == 'F':
            enc=1
        elif sex==2 and data.sex_train[i] == 'M':
            enc=1
        else:
            enc=0
        
        if enc==1:
            img_label = labels[np.argmax(data.y_train[i])]
            if(img_label == label):
                # Use the z mean for VAE models, in other case use all z
                if vae:
                    neighbor_list = get_nn_padchest(lat_rep[0], img_index=i)
                else:
                    neighbor_list = get_nn_padchest(lat_rep, img_index=i)

                if age!=0:
                    plot_nn_padchest(ax[cont], neighbor_list, data, i, n_neighbors,
                                     show_age=True, show_sex=True)
                elif sex!=0:
                    plot_nn_padchest(ax[cont], neighbor_list, data, i, n_neighbors,
                                     show_age=True, show_sex=True)
                else:
                    print('Error, using stratified without an unique objective')

                cont = cont+1

        i = i+1
    
    plt.savefig(path)
    plt.show()


# BORRAR, solo es util para generar imagenes con los resultados
def plot_nneighbors_filtered_original_stratify(path, data, model, n_neighbors=5, n_cases=5, vae=False, label="normal",
                                               age=0, sex=0):
    cont = 0
    i = 0
    
    lat_rep = model.predict(data.X_train)
    
    f, ax = plt.subplots(n_cases, n_neighbors + 1, figsize=(15*n_neighbors, 15*n_cases), dpi=100)
    plt.rc('font', size=25)
        
    labels = sorted(config["padchest"]["label_names"], key=config["padchest"]["label_names"].get)
    
    while cont<n_cases and i<len(data.X_train):
        if age==1 and data.age_train[i] >= 65:
            enc=1
        elif age==2 and data.age_train[i] < 65:
            enc=1
        elif sex==1 and data.sex_train[i] == 'F':
            enc=1
        elif sex==2 and data.sex_train[i] == 'M':
            enc=1
        else:
            enc=0
        
        if enc==1:
            img_label = labels[np.argmax(data.y_train[i])]

            if(img_label == label):
                # Use the z mean for VAE models, in other case use all z
                if vae:
                    neighbor_list = get_nn_padchest(lat_rep[0], img_index=i)
                else:
                    neighbor_list = get_nn_padchest(lat_rep, img_index=i)

                plot_nn_padchest_save(ax[cont], neighbor_list, data, i, n_neighbors,
                                      show_age=age, show_sex=sex)

                cont = cont+1

        i = i+1
    
    plt.savefig(path)
    plt.show()


def get_recom_results(data, model, n_imgs, n_neighbors, verbose=False, vae=False):
    i = 0
    results = []
    
    with open('/workspace/Guille/MOC-AE/MOC-AE_Code/config.json', 'r') as f:
        config = json.load(f)
    labels = sorted(config["padchest"]["label_names"], key=config["padchest"]["label_names"].get)
    
    lat_rep = model.predict(data.X_val)
    
    while i<n_imgs and i<len(data.X_val):
        img_label = labels[np.argmax(data.y_val[i])]
        
        # Use the z mean for VAE models, in other case use all z
        if vae:
            neighbor_list = get_nn_padchest(lat_rep[0], img_index=i)
        else:
            neighbor_list = get_nn_padchest(lat_rep, img_index=i)
            
        labels_neigbor = []

        for j in range(n_neighbors):
            labels_neigbor.append(labels[np.argmax(data.y_val[neighbor_list[j][1]])])

        results.append([img_label, labels_neigbor])
        
        i = i+1
        if i%25==0 and verbose:
            print(f"{i} images processed")
            
    return results


def get_recom_results_filtered(X, y, names, model, n_imgs, n_neighbors, verbose=False, vae=False):
    cont = 0
    i = 0
    
    results = []
    labels = sorted(config["padchest"]["label_names"], key=config["padchest"]["label_names"].get)
    
    lat_rep = model.predict(data.X_val)
    
    while cont<n_imgs and i<len(data.X_val):
        img_label = labels[np.argmax(data.y_val[i])]
        if(img_label != "normal"):
            # Use the z mean for VAE models, in other case use all z
            if vae:
                neighbor_list = get_nn_padchest(lat_rep[0], img_index=i)
            else:
                neighbor_list = get_nn_padchest(lat_rep, img_index=i)
            
            labels_neigbor = []
            
            for j in range(n_neighbors):
                labels_neigbor.append(labels[np.argmax(data.y_val[neighbor_list[j][1]])])
                
            results.append([img_label, labels_neigbor])
            
            cont = cont + 1
        
        i = i+1
        
        if i%100==0 and verbose:
            print(f"{i} images readed. {cont} not normal images processed")
            
    return results


def get_recom_results_normal(data, model, label_base, n_imgs, n_neighbors,
                            verbose=False, vae=False):
    cont = 0
    i = 0
    
    results = []
    labels = sorted(config["padchest"]["label_names"], key=config["padchest"]["label_names"].get)
    
    lat_rep = model.predict(data.X_val)
    
    while cont<n_imgs and i<len(data.X_val):
        img_label = labels[np.argmax(data.y_val[i])]
        if img_label == label_base:
            # Use the z mean for VAE models, in other case use all z
            if vae:
                neighbor_list = get_nn_padchest_filter_normal(data, labels, lat_rep[0], label_base, img_index=i)
            else:
                neighbor_list = get_nn_padchest_filter_normal(data, labels, lat_rep, label_base, img_index=i)
            
            labels_neigbor = []
            
            for j in range(n_neighbors):
                labels_neigbor.append(labels[np.argmax(data.y_val[neighbor_list[j][1]])])
                
            results.append([img_label, labels_neigbor])
            
            cont = cont + 1
        
        i = i+1
        
        if i%100==0 and verbose:
            print(f"{i} images readed. {cont} not normal images processed")
            
    return results


def get_recom_results_stratify(data, model, n_imgs, n_neighbors,
                               verbose=False, vae=False, age=0, sex=0):
    cont = 0
    i = 0
    
    results = []
    labels = sorted(config["padchest"]["label_names"], key=config["padchest"]["label_names"].get)
    
    lat_rep = model.predict(data.X_val)
    
    while cont<n_imgs and i<len(data.X_val):
        if age==1 and data.age_train[i] >= 65:
            enc=1
        elif age==2 and data.age_train[i] < 65:
            enc=1
        elif sex==1 and data.sex_train[i] == 'F':
            enc=1
        elif sex==2 and data.sex_train[i] == 'M':
            enc=1
        else:
            enc=0
        
        if enc==1:
            img_label = labels[np.argmax(data.y_val[i])]
            # Use the z mean for VAE models, in other case use all z
            if vae:
                neighbor_list = get_nn_padchest(lat_rep[0], img_index=i)
            else:
                neighbor_list = get_nn_padchest(lat_rep, img_index=i)

            labels_neigbor = []

            for j in range(n_neighbors):
                labels_neigbor.append(labels[np.argmax(data.y_val[neighbor_list[j][1]])])

            results.append([img_label, labels_neigbor])

            cont = cont + 1
        
        i = i+1
        
        if i%100==0 and verbose:
            print(f"{i} images readed. {cont} not normal images processed")
            
    return results


def print_recom_results(results):
    i=0

    for elem in results:
        y_real = elem[0]
        y_pred = elem[1]

        correct_pred = y_pred.count(y_real)
        pred_acc = (correct_pred / len(y_pred)) * 100

        print(f"Etiqueta objetivo: {y_real}")
        print(f"Predicciones: {y_pred}")
        print(f"Porcentaje de precisión: {pred_acc}%")

        if i==10:
            break
        i = i+1


def get_recom_acc(results):
    pred_acc = []

    for elem in results:
        y_real = elem[0]
        y_pred = elem[1]

        correct_pred = y_pred.count(y_real)
        pred_acc.append((correct_pred / len(y_pred)) * 100)

    return sum(pred_acc) / len(pred_acc), np.std(pred_acc)


def get_recom_rec(results):
    pred_acc = []

    for elem in results:
        y_real = elem[0]
        y_pred = elem[1]

        correct_pred = y_pred.count(y_real)
        pred_acc.append((correct_pred / len(y_pred)) * 100)

    return sum(pred_acc) / len(pred_acc), np.std(pred_acc)


def get_dist_acc(results):
    pred_acc = []

    for elem in results:
        pred_acc.append(mean(elem))

    return mean(pred_acc), stdev(pred_acc)


def sliced_wasserstein(X, Y, num_proj):
    # Code adapted from https://gist.github.com/smestern/ba9ee191ca132274c4dfd6e1fd6167ac
    dim = X.shape[1]
    ests = []
    for _ in range(num_proj):
        # sample uniformly from the unit sphere
        dir_proj = np.random.rand(dim)
        dir_proj /= np.linalg.norm(dir_proj)

        # project the data
        X_proj = X @ dir_proj
        Y_proj = Y @ dir_proj

        # compute 1d wasserstein
        ests.append(wasserstein_distance(X_proj, Y_proj))
    return np.mean(ests)


def wass_distance(img1, img2):
    img1 = cv2.resize(img1, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    img1_eq = exposure.equalize_hist(img1)
    
    img2 = cv2.resize(img2, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    img2_eq = exposure.equalize_hist(img2)
    
    return sliced_wasserstein(img1_eq[:,:], img2_eq[:,:], 100)


def get_distance_results(data, model, n_imgs, n_neighbors, verbose=False, vae=False):
    i = 0
    
    results = []
    
    labels = sorted(config["padchest"]["label_names"], key=config["padchest"]["label_names"].get)
    
    lat_rep = model.predict(data.X_val)
    
    while i<n_imgs and i<len(data.X_val):
        
        # Use the z mean for VAE models, in other case use all z
        if vae:
            neighbor_list = get_nn_padchest(lat_rep[0], img_index=i)
        else:
            neighbor_list = get_nn_padchest(lat_rep, img_index=i)
            
        labels_neigbor = []
            
        for j in range(n_neighbors):
            labels_neigbor.append(wass_distance(data.X_val[i], data.X_val[neighbor_list[j][1]]))

        results.append(labels_neigbor)
        
        i = i+1
        if i%25==0 and verbose:
            print(f"{i} images processed")
            
    return results


def get_distance_results_normal(data, model, label_base, n_imgs, n_neighbors,
                                verbose=False, vae=False):
    cont = 0
    i = 0
    
    results = []
    labels = sorted(config["padchest"]["label_names"], key=config["padchest"]["label_names"].get)
    
    lat_rep = model.predict(data.X_val)
    
    while cont<n_imgs and i<len(data.X_val):
        img_label = labels[np.argmax(data.y_val[i])]
        if img_label == label_base:
            # Use the z mean for VAE models, in other case use all z
            if vae:
                neighbor_list = get_nn_padchest_filter_normal(data, labels, lat_rep[0], label_base, img_index=i)
            else:
                neighbor_list = get_nn_padchest_filter_normal(data, labels, lat_rep, label_base, img_index=i)

            distances = []

            for j in range(n_neighbors):
                distances.append(wass_distance(data.X_val[i], data.X_val[neighbor_list[j][1]]))

            results.append(distances)
            cont+=1
        
        i = i+1
        if i%25==0 and verbose:
            print(f"{i} images processed")
            
    return results


def get_distance_results_stratify(data, model, n_imgs, n_neighbors,
                                verbose=False, vae=False, age=0, sex=0):
    cont = 0
    i = 0
    
    results = []
    labels = sorted(config["padchest"]["label_names"], key=config["padchest"]["label_names"].get)
    
    lat_rep = model.predict(data.X_val)
    
    while cont<n_imgs and i<len(data.X_val):
        if age==1 and data.age_train[i] >= 65:
            enc=1
        elif age==2 and data.age_train[i] < 65:
            enc=1
        elif sex==1 and data.sex_train[i] == 'F':
            enc=1
        elif sex==2 and data.sex_train[i] == 'M':
            enc=1
        else:
            enc=0
        
        if enc==1:
            img_label = labels[np.argmax(data.y_val[i])]
            # Use the z mean for VAE models, in other case use all z
            if vae:
                neighbor_list = get_nn_padchest(lat_rep[0], img_index=i)
            else:
                neighbor_list = get_nn_padchest(lat_rep, img_index=i)

            distances = []

            for j in range(n_neighbors):
                distances.append(wass_distance(data.X_val[i], data.X_val[neighbor_list[j][1]]))

            results.append(distances)
            cont+=1
        
        i = i+1
        if i%25==0 and verbose:
            print(f"{i} images processed")
            
    return results
