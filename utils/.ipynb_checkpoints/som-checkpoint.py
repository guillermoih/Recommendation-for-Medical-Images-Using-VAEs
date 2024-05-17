# +
import json
import torch
import time

import matplotlib.pyplot as plt
import pandas as pd

from scipy.spatial.distance import euclidean
from geomloss import SamplesLoss
from utils.plot_basic import *
from utils.recommendation import *
# -

with open('/workspace/Guille/MOC-AE/MOC-AE_Code/config.json', 'r') as f:
    config = json.load(f)


# # SOM utilities

def get_heatmap(img, som):
    heat = np.empty([som.distance_map().shape[0], som.distance_map().shape[1]])

    for i in range(som.distance_map().shape[0]):
        for j in range(som.distance_map().shape[1]):
            heat[i][j] = euclidean(img, som.get_weights()[i][j])
            
    return heat


def get_heatmaps(dataset, som):
    heatmaps = []
    
    for i in dataset:
        heatmaps.append(get_heatmap(i.flatten(), som))
        
    return heatmaps


def get_nn_df(query, query_heatmap, X, X_heatmap, sort_by='heatmap_dist'):
    Loss = SamplesLoss("sinkhorn", blur=0.05,)
    
    df = pd.DataFrame(columns=['img', 'img_dist', 'heatmap_dist'])
    
    for i in range(len(X_heatmap)):
        it_img = X[i].reshape(config["padchest"]["image"]["img_height"],
                              config["padchest"]["image"]["img_width"])
        
        if sort_by=="both":
            heatmap_distance = sliced_wasserstein(query_heatmap, X_heatmap[i], 100)
            img_distance = sliced_wasserstein(query, torch.from_numpy(it_img), 100)
        
        elif sort_by=="heatmap_dist":
            #heatmap_distance = Loss(torch.from_numpy(query_heatmap), torch.from_numpy(X_heatmap[i])).item()
            heatmap_distance = sliced_wasserstein(query_heatmap, X_heatmap[i], 100)
            img_distance = -1
            
        else:
            #img_distance = Loss(torch.from_numpy(query), torch.from_numpy(it_img)).item()
            img_distance = sliced_wasserstein(query, torch.from_numpy(it_img), 100)
            heatmap_distance = -1
        
        row = pd.Series({'img': i,
                         'img_dist': img_distance,
                         'heatmap_dist': heatmap_distance},
                        name=i)
        
        df = df.append(row)
        
    return df


def get_recom_results_som(X, y, names, som, n_imgs, k=5, sort_by='heatmap_dist'):
    print("Calculating all heatmaps")
    print(time.strftime("%H:%M:%S", time.localtime()))
    heatmaps = get_heatmaps(X, som)
    
    i = 0
    results = []
    
    while i<n_imgs and i<len(X):        
        query_img = X[i].reshape(config["padchest"]["image"]["img_height"],
                              config["padchest"]["image"]["img_width"])
        query_colors = heatmaps[i]
        
        img_label = names[np.argmax(y[i])]
        
        df_neighbors = get_nn_df(query_img, query_colors, X, heatmaps)
        df_neighbors = df_neighbors.sort_values(by=[sort_by])

        labels_neigbor = []

        for j in range(1, k+1):
            img_row = df_neighbors.iloc[j]
            img_idx = int(img_row['img'])
            
            labels_neigbor.append(names[np.argmax(y[img_idx])])

        results.append([img_label, labels_neigbor])
        
        i = i+1
        if i%25==0:
            print(f"{i} images processed")
            
    return results


def get_recom_results_som_mocae(X, lat_rep, y, names, som, n_imgs, k=5, sort_by='heatmap_dist'):
    print("Calculating all heatmaps")
    print(time.strftime("%H:%M:%S", time.localtime()))
    heatmaps = get_heatmaps(lat_rep, som)
    
    i = 0
    results = []
    
    while i<n_imgs and i<len(X):        
        query_img = X[i].reshape(config["padchest"]["image"]["img_height"],
                              config["padchest"]["image"]["img_width"])
        query_colors = heatmaps[i]
        
        img_label = names[np.argmax(y[i])]
        
        df_neighbors = get_nn_df(query_img, query_colors, X, heatmaps)
        df_neighbors = df_neighbors.sort_values(by=[sort_by])

        labels_neigbor = []

        for j in range(1, k+1):
            img_row = df_neighbors.iloc[j]
            img_idx = int(img_row['img'])
            
            labels_neigbor.append(names[np.argmax(y[img_idx])])

        results.append([img_label, labels_neigbor])
        
        i = i+1
        if i%25==0:
            print(f"{i} images processed")
            
    return results


def get_distance_results_som_mocae(X, lat_rep, y, names, som, n_imgs, k):
    i = 0
    results = []
    
    while i<n_imgs and i<len(X):
        query_img = X[i].reshape(config["padchest"]["image"]["img_height"],
                              config["padchest"]["image"]["img_width"])
        
        query_colors = heatmaps[i]
        
        df_neighbors = get_nn_df(query_img, query_colors, X, heatmaps, sort_by="both")
        df_neighbors = df_neighbors.sort_values(by=[sort_by])

        labels_neigbor = []

        for j in range(1, k+1):
            img_row = df_neighbors.iloc[j]
            img_idx = int(img_row['img'])
            
            labels_neigbor.append(wass_distance(query_img, X[img_idx]))

        results.append(labels_neigbor)
        
        i = i+1
        if i%25==0:
            print(f"{i} images processed")
            
    return results


# # Plotting functions

def plot_img_heatmaps(X, y, names, som, query_id=0, n_imgs=5):
    Loss = SamplesLoss("sinkhorn", blur=0.05,)
    fig, ax = plt.subplots(2, n_imgs+1, figsize=(4*(n_imgs+1), 8))

    query_img = X[query_id].reshape(config["padchest"]["image"]["img_height"],
                                    config["padchest"]["image"]["img_width"])

    show_case(X, y, names, query_id, ax[0][0])
    ax[0][0].set_title("Query image")

    query_colors = get_heatmap(query_img.flatten(), som)
    ax[1][0].set_title("Query heatmap")
    ax[1][0].imshow(query_colors)

    for i in range(query_id+1, query_id+n_imgs+1):
        it_img = X[i].reshape(config["padchest"]["image"]["img_height"],
                              config["padchest"]["image"]["img_width"])

        img_distance = Loss(torch.from_numpy(query_img), torch.from_numpy(it_img)).item()

        show_case(X, y, names, i, ax[0][i-query_id])
        ax[0][i-query_id].set_title("Distance:\n" + str(img_distance))

        it_colors = get_heatmap(it_img.flatten(), som)
        heatmap_distance = Loss(torch.from_numpy(query_colors), torch.from_numpy(it_colors)).item()

        ax[1][i-query_id].set_title("Heatmap distance:\n" + str(heatmap_distance))
        ax[1][i-query_id].imshow(it_colors)

    plt.show()


def plot_latent_winner(X, y, names, som, n_cases = 5, idx_ini = 0):
    fig, ax = plt.subplots(2, n_cases, figsize=(n_cases*3, 6))

    ax[0][0].set_title("Input image")
    ax[1][0].set_title("Latent weights")

    for i in range(idx_ini, idx_ini + n_cases):
        show_case(X, y, names, i, ax[0][i - idx_ini])
        show_case(X, y, names, i, ax[0][i - idx_ini])

        ax[1][i - idx_ini].imshow(som.get_weights()[som.winner(X[i].flatten())]
                                  .reshape(config["padchest"]["image"]["img_height"],
                                           config["padchest"]["image"]["img_width"]),
                                  cmap='gray')
        
        ax[1][i - idx_ini].axis("off")

    plt.show()


def plot_latent_heatmap(som):
    plt.figure(figsize=(9, 9))

    plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
    plt.colorbar()

    plt.show()


def plot_latent_imgmap(som):
    fig, ax = plt.subplots(som.distance_map().shape[0], som.distance_map().shape[1],
                           figsize=(som.distance_map().shape[0]*3, som.distance_map().shape[1]*3))

    for i in range(som.distance_map().shape[0]):
        for j in range(som.distance_map().shape[1]):
            ax[i][j].imshow(som.get_weights()[i][j]
                            .reshape(config["padchest"]["image"]["img_height"],
                                     config["padchest"]["image"]["img_width"]),
                            cmap='gray')
            
            ax[i][j].axis("off")
            
    plt.show()


def plot_neighborhood(X, y, names, som, k=5, n_examples=1, sort_by='heatmap_dist'):
    fig, ax = plt.subplots(2*n_examples, k+1, figsize=(4*(k+1), 9*n_examples))
    
    print("Calculating all heatmaps")
    print(time.strftime("%H:%M:%S", time.localtime()))
    heatmaps = get_heatmaps(X, som)

    for i in range(n_examples):
        print("Getting neighbors for img ", i)
        print(time.strftime("%H:%M:%S", time.localtime()))
        
        query_img = X[i].reshape(config["padchest"]["image"]["img_height"],
                                 config["padchest"]["image"]["img_width"])
        query_colors = heatmaps[i]

        df_neighbors = get_nn_df(query_img, query_colors, X, heatmaps)
        df_neighbors = df_neighbors.sort_values(by=[sort_by])

        show_case(X, y, names, i, ax[i*2][0])
        ax[i*2][0].set_title("Query image")

        ax[i*2 + 1][0].set_title("Query heatmap")
        ax[i*2 + 1][0].imshow(query_colors)

        for j in range(1, k+1):
            img_row = df_neighbors.iloc[j]
            img_idx = int(img_row['img'])
            img_it = X[img_idx].reshape(config["padchest"]["image"]["img_height"],
                                        config["padchest"]["image"]["img_width"])

            show_case(X, y, names, img_idx, ax[i*2][j])
            ax[i*2][j].set_title("Distance:\n" + str(img_row['img_dist']))

            ax[i*2 + 1][j].set_title("Heatmap distance:\n" + str(img_row['heatmap_dist']))
            ax[i*2 + 1][j].imshow(heatmaps[img_idx])
            
    plt.show()


def plot_neighborhood_mocae(X, lat_rep, y, names, som, k=5, n_examples=1, sort_by='heatmap_dist'):
    fig, ax = plt.subplots(2*n_examples, k+1, figsize=(4*(k+1), 9*n_examples))
    
    print("Calculating all heatmaps")
    print(time.strftime("%H:%M:%S", time.localtime()))
    heatmaps = get_heatmaps(lat_rep, som)

    for i in range(n_examples):
        print("Getting neighbors for img ", i)
        print(time.strftime("%H:%M:%S", time.localtime()))
        
        query_img = X[i].reshape(config["padchest"]["image"]["img_height"],
                                 config["padchest"]["image"]["img_width"])
        query_colors = heatmaps[i]

        df_neighbors = get_nn_df(query_img, query_colors, X, heatmaps)
        df_neighbors = df_neighbors.sort_values(by=[sort_by])

        show_case(X, y, names, i, ax[i*2][0])
        ax[i*2][0].set_title("Query image")

        ax[i*2 + 1][0].set_title("Query heatmap")
        ax[i*2 + 1][0].imshow(query_colors)

        for j in range(1, k+1):
            img_row = df_neighbors.iloc[j]
            img_idx = int(img_row['img'])
            img_it = X[img_idx].reshape(config["padchest"]["image"]["img_height"],
                                        config["padchest"]["image"]["img_width"])

            show_case(X, y, names, img_idx, ax[i*2][j])
            ax[i*2][j].set_title("Distance:\n" + str(img_row['img_dist']))

            ax[i*2 + 1][j].set_title("Heatmap distance:\n" + str(img_row['heatmap_dist']))
            ax[i*2 + 1][j].imshow(heatmaps[img_idx])
            
    plt.show()
