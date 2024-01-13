import pandas as pd
import matplotlib
matplotlib.use("agg")
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import numpy as np
from numpy import stack, zeros, unique, arange, log, vectorize, logical_or
from numpy import mean as np_mean
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from itertools import combinations
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import os

from evaluate_data_preprocess import evaluate_dataset_load

def get_knn_purity(latents, labels, n_neighs=30):

    nbrs = NearestNeighbors(n_neighbors=n_neighs + 1).fit(latents)
    indices = nbrs.kneighbors(latents, return_distance=False)[:, 1:]
    neigh_labels = vectorize(lambda x: labels[x])(indices)

    scores = ((neigh_labels - labels.reshape(-1, 1)) == 0).mean(axis=1)
    res = [np_mean(scores[labels.reshape(scores.shape) == i]) for i in unique(labels)]

    return np_mean(res)

def get_normalized_mutual_info_score(latents, labels, num_cluster = 2):

    #predict_labels = KMeans(n_clusters=num_cluster, random_state=42).fit_predict(latents)
    predict_labels = GaussianMixture(n_components=num_cluster, random_state=42).fit_predict(latents)
    #predict_labels = AgglomerativeClustering(n_clusters=num_cluster).fit_predict(latents)
    #predict_labels = DBSCAN().fit_predict(latents)
    nmi = normalized_mutual_info_score(labels, predict_labels)

    return nmi

def get_adjusted_rand_score(latents, labels, num_cluster=2):

    #predict_labels = KMeans(n_clusters=num_cluster, random_state=42).fit_predict(latents)
    predict_labels = GaussianMixture(n_components=num_cluster, random_state=42).fit_predict(latents)
    #predict_labels = AgglomerativeClustering(n_clusters=num_cluster).fit_predict(latents)
    #predict_labels = DBSCAN().fit_predict(latents)
    adj_rand_index = adjusted_rand_score(labels, predict_labels)

    return adj_rand_index

def get_silhouette_score(latents, num_cluster=2):
    #predict_labels = KMeans(n_clusters=num_cluster, random_state=42).fit_predict(latents)
    predict_labels = GaussianMixture(n_components=num_cluster, random_state=42).fit_predict(latents)
    #predict_labels = AgglomerativeClustering(n_clusters=num_cluster).fit_predict(latents)
    sil_score = silhouette_score(latents, predict_labels)
    return sil_score

def complete_diameter_distance(X):
    res = []
    for i, j in combinations(range(X.shape[0]), 2):
        a_i = X[i, :]
        a_j = X[j, :]
        res.append(np.linalg.norm(a_i - a_j))
    return np.max(res)

def average_of_centroids_linkage_distance(X1, X2):
    center1 = np.mean(X1, axis=0)
    center2 = np.mean(X2, axis=0)
    res = []
    for x1 in X1:
        res.append(np.linalg.norm(x1 - center2))
    for x2 in X2:
        res.append(np.linalg.norm(x2 - center1))
    return np.mean(res)

def get_Dunn_index(latents, num_cluster=2):
    #predict_labels = KMeans(n_clusters=num_cluster, random_state=42).fit_predict(latents)
    predict_labels = GaussianMixture(n_components=num_cluster, random_state=42).fit_predict(latents)
    #predict_labels = AgglomerativeClustering(n_clusters=num_cluster).fit_predict(latents)
    # get minimum value of inter_cluster_distance
    res1 = []
    for i, j in combinations(np.unique(predict_labels), 2):
        X1 = latents[np.where(predict_labels == i)[0], :]
        X2 = latents[np.where(predict_labels== j)[0], :]
        res1.append(average_of_centroids_linkage_distance(X1, X2))
    min_inter_cd = np.min(res1)
    # get maximum value of intra_cluser_distance
    res2 = []
    for label in np.unique(predict_labels):
        X_target = latents[np.where(predict_labels == label)[0], :]
        if X_target.shape[0] >= 2:
            res2.append(complete_diameter_distance(X_target))
        else:
            res2.append(0)
    max_intra_cd = np.max(res2)
    Dunn_idx = min_inter_cd / max_intra_cd
    return Dunn_idx

def get_mse_score(real_latents, reconstruction_latents):
    mse = (reconstruction_latents - real_latents) ** 2
    return np.mean(np.mean(mse))

def clustering_evaluation(cancer_type = ['COAD'], significant_gene_num  = 300, train_epoch = 500,augmentation= 'yes'):

    f_name = f_name = './../clustering_evaluation/'
    os.makedirs(f_name, exist_ok=True)

    cluster_eval_df = pd.DataFrame()

    for cancer in cancer_type:
        #print("cancer type is %s"%cancer)
        Final_TCGA, fake_A, reconstruct_A, Final_TCGA2, fake_B, reconstruct_B = evaluate_dataset_load(cancer=cancer,DIR='./../saved_models/',significant_gene_num=significant_gene_num,train_epoch=train_epoch)
        Final_TCGA = Final_TCGA.iloc[:,0:significant_gene_num+1]
        Final_TCGA2 = Final_TCGA2.iloc[:, 0:significant_gene_num + 1]

        if augmentation=='no':
            Total_TCGA = pd.concat([Final_TCGA.reset_index(drop=True), Final_TCGA2.reset_index(drop=True)], axis=0)
            Total_TCGA = Total_TCGA.reset_index(drop=True)

            string_1 = 1
            string_2 = 0

            Total_tcga_real_label = []
            for i in range(Final_TCGA.shape[0]):
                Total_tcga_real_label.append(string_1)
            for i in range(Final_TCGA2.shape[0]):
                Total_tcga_real_label.append(string_2)

        else:
            Total_TCGA = pd.concat([fake_A, fake_B, Final_TCGA.reset_index(drop=True), Final_TCGA2.reset_index(drop=True)], axis=0)
            Total_TCGA = Total_TCGA.reset_index(drop=True)

            string_1 = 1
            string_2 = 0
            string_3 = 1
            string_4 = 0

            Total_tcga_real_label = []
            for i in range(fake_A.shape[0]):
                Total_tcga_real_label.append(string_1)
            for i in range(fake_B.shape[0]):
                Total_tcga_real_label.append(string_2)
            for i in range(Final_TCGA.shape[0]):
                Total_tcga_real_label.append(string_3)
            for i in range(Final_TCGA2.shape[0]):
                Total_tcga_real_label.append(string_4)


        Total_TCGA = np.asarray(Total_TCGA)
        Total_tcga_real_label = np.asarray(Total_tcga_real_label)

        tsne = TSNE(random_state=42, perplexity=50)
        Total_tsne = tsne.fit_transform(Total_TCGA[:, 0:300])
        tsne_results = pd.DataFrame(Total_tsne, columns=['tsne1', 'tsne2']).assign(category=Total_tcga_real_label).groupby('category')
        for name, points in tsne_results:
            if name == 0:
                X1 = points.iloc[:, 0:2]
            else:
                X2 = points.iloc[:, 0:2]

        Total_TCGA = pd.concat([X1.reset_index(drop=True), X2.reset_index(drop=True)], axis=0)
        Total_TCGA = Total_TCGA.reset_index(drop=True)
        string_1 = 0
        string_2 = 1
        Total_tcga_real_label = []
        for i in range(X1.shape[0]):
            Total_tcga_real_label.append(string_1)
        for i in range(X2.shape[0]):
            Total_tcga_real_label.append(string_2)
        Total_TCGA = np.asarray(Total_TCGA)
        Total_tcga_real_label = np.asarray(Total_tcga_real_label)

        length = len(Total_TCGA)
        indices = np.array([i for i in range(length)])
        np.random.seed(42)
        np.random.shuffle(indices)

        Total_TCGA_shuffle = Total_TCGA[indices]
        Total_tcga_real_label_shuffle = Total_tcga_real_label[indices]

        knn_purity_value = get_knn_purity(Total_TCGA_shuffle[:,0:300], Total_tcga_real_label_shuffle, n_neighs=30)
        nmi_score = get_normalized_mutual_info_score(Total_TCGA_shuffle[:,0:300], Total_tcga_real_label_shuffle, num_cluster=2)
        adj_rand_score = get_adjusted_rand_score(Total_TCGA_shuffle[:,0:300], Total_tcga_real_label_shuffle, num_cluster=2)
        silhouette_score_value = get_silhouette_score(Total_TCGA_shuffle[:,0:300], num_cluster=2)
        dunn_index_value = get_Dunn_index(Total_TCGA_shuffle[:,0:300], num_cluster=2)

        if augmentation=='no':
            mse_value = 'NA'
            r2_value = 'NA'
        else:
            Final_TCGA.reset_index(drop=True, inplace=True)
            mse_value = get_mse_score(Final_TCGA.iloc[:,0:300], reconstruct_A.iloc[:,0:300])
            r2_value = (Final_TCGA.iloc[:,0:300].corrwith(reconstruct_A.iloc[:,0:300])**2).mean()

        knn_purity_value = round(knn_purity_value,3)
        nmi_score = round(nmi_score, 3)
        adj_rand_score = round(adj_rand_score, 3)
        silhouette_score_value = round(silhouette_score_value,3)
        dunn_index_value = round(dunn_index_value, 3)

        if augmentation!='no':
            mse_value = round(mse_value, 3)
            r2_value = round(r2_value, 3)

        dic = { 'cancer type': [cancer],'significant gene num': [significant_gene_num], 'train epoch': [train_epoch], 'augmentation type': [augmentation], 'KNN purity': [knn_purity_value],'NMI': [nmi_score], 'adjusted rand score': [adj_rand_score],'Sihouette score': [silhouette_score_value], 'Dunn index': [dunn_index_value], 'mse': [mse_value], 'R2': [r2_value]}
        dic_df = pd.DataFrame.from_dict(dic)

        cluster_eval_df = pd.concat([cluster_eval_df, dic_df], axis=0)

        os.makedirs(f_name + 'gene_' + str(significant_gene_num) + '_' + 'epoch_'+ str(train_epoch) +'/', exist_ok=True)
        file_name = f_name + 'gene_' + str(significant_gene_num) + '_' + 'epoch_'+ str(train_epoch) +'/' + augmentation+ '_clustering evaluation.csv'
        cluster_eval_df.to_csv(file_name, index=False, header=True)
