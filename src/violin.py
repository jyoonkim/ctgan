import os
import pandas as pd
import matplotlib
matplotlib.use("agg")
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from evaluate_data_preprocess import evaluate_dataset_load

def violin_plot(cancer_type = ['COAD'], significant_gene_num  = 300, train_epoch = 500):
    #cancer_type = ['COAD']  ## If you want all cancer types -> ['BLCA', 'COAD','HNSC', 'KIRC', 'LGG','LIHC', 'LUAD', 'LUSC', 'OV', 'SKCM', 'STAD']

    ensemble_gene_symbol = pd.read_csv('./../data/ensemble2genesymbol.csv', sep=',')

    for cancer in cancer_type:

        Final_TCGA, fake_A, reconstruct_A, Final_TCGA2, fake_B, reconstruct_B = evaluate_dataset_load(cancer=cancer,DIR='./../saved_models/',significant_gene_num=significant_gene_num,train_epoch=train_epoch)

        f_name = './../violin_plot_evaluation/' + cancer + '/' +  'gene_' + str(significant_gene_num) + '_' + 'epoch_' + str(train_epoch) + '/'

        os.makedirs(f_name, exist_ok=True)

        #### Final_TCGA (real data)  reconstruct_A (reconstructed data) ####
        Final_TCGA = Final_TCGA.iloc[:,0:significant_gene_num+1]

        Final_TCGA_mean = Final_TCGA.mean()
        Final_TCGA_std = Final_TCGA.std()

        reconstruct_A_mean = reconstruct_A.mean()
        reconstruct_A_std = reconstruct_A.std()

        for gene_num in [0,1,2,3,4,5,6,7,8,9,-1]:
            Final_TCGA_gene = Final_TCGA.iloc[:, gene_num]
            reconstruct_A_gene = reconstruct_A.iloc[:, gene_num]
            gene_df = pd.concat([Final_TCGA_gene, reconstruct_A_gene], axis=0)
            gene_df = gene_df.reset_index(drop=True)
            gene_df = pd.DataFrame(gene_df)

            type = np.concatenate((np.repeat("real", Final_TCGA.shape[0]).reshape(-1, 1),
                                   np.repeat("reconstructed", reconstruct_A.shape[0]).reshape(-1, 1)), axis=0)
            type = pd.DataFrame(type)

            if gene_num!=-1:
                total_df = pd.concat([gene_df, type], axis=1)
                ensemble_id = total_df.columns.to_list()[0]
                gene_name = ensemble_gene_symbol[ensemble_gene_symbol['ensemble_id'] == ensemble_id]['gene_name'].to_list()[0]
                total_df.columns = ['gene expression', 'type']
                plt.figure(figsize=(12, 8))
                ax = sns.violinplot(x='type', y='gene expression', data=total_df)
                plt.ylabel(gene_name, fontsize=30)
            else:
                total_df = pd.concat([gene_df, type], axis=1)
                total_df.columns = ['days', 'type']
                plt.figure(figsize=(12, 8))
                ax = sns.violinplot(x='type', y='days', data=total_df)
                plt.ylabel('Days', fontsize=30)

            plt.xlabel("", fontsize=33)

            plt.xticks(fontsize=33)
            plt.yticks(fontsize=25)

            if gene_num!=-1:
                plt.savefig(f_name + '/gene_' + str(gene_num) + ".tiff", dpi=350)
            else:
                plt.savefig(f_name + '/days.tiff', dpi=350)

