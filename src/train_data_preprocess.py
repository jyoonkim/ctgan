import pandas as pd
import numpy as np
import pickle

def train_dataset_load(cancer_type = 'COAD', significant_gene_num = 300):
    Surv_data2 = pd.read_csv('./../data/survival_data/event_days_' + 'BRCA'+ '.csv', sep=',', header=0, index_col=0)

    with open('./../data/preprocess_ge_data/TCGA-BRCA_preprocess.txt', 'rb') as file:
        TCGA2 = pickle.load(file)

    with open('./../data/significant_gene/'+ cancer_type+'/'+str(significant_gene_num)+'_Survival_related_significant_gene_list','rb') as f:
        significant_genes = pickle.load(f)


    TCGA2 = TCGA2.iloc[:, significant_genes]
    TCGA2_gene = TCGA2

    TCGA2_gene = np.log2(TCGA2_gene + 1)
    ##############################################################
    patients = Surv_data2.index.tolist()

    mrna_header = TCGA2_gene.index.to_list()

    days_list = []
    event_list = []

    split_mrna_header_list = []

    for i in mrna_header:
        split_mrna_header = i[0:12]
        split_mrna_header_list.append(split_mrna_header)

    for i in split_mrna_header_list:
        days = Surv_data2.loc[i]['days']
        event = Surv_data2.loc[i]['event']
        days = np.log2(days + 1)
        days_list.append(days)
        event_list.append(event)

    TCGA2_gene["days"] = days_list
    TCGA2_gene["event"] = event_list

    TCGA2_gene = TCGA2_gene[TCGA2_gene['days'] != 0]

    TCGA2_gene = TCGA2_gene.iloc[:, 0:-1]

    Final_TCGA2 = TCGA2_gene

    ##################################################################################
    Surv_data1 = pd.read_csv('./../data/survival_data/event_days_' + cancer_type+ '.csv', sep=',', header=0, index_col=0)

    with open('./../data/preprocess_ge_data/TCGA-'+cancer_type+'_preprocess.txt', 'rb') as file:
        TCGA = pickle.load(file)

    with open('./../data/significant_gene/'+cancer_type+'/'+str(significant_gene_num)+'_Survival_related_significant_gene_list','rb') as f:
        significant_genes = pickle.load(f)

    TCGA = TCGA.iloc[:, significant_genes]
    TCGA_gene = TCGA

    TCGA_gene = np.log2(TCGA_gene + 1)

    ##############################################################
    patients = Surv_data1.index.tolist()

    mrna_header = TCGA_gene.index.to_list()

    days_list = []
    event_list = []

    split_mrna_header_list = []

    for i in mrna_header:
        split_mrna_header = i[0:12]
        split_mrna_header_list.append(split_mrna_header)

    for i in split_mrna_header_list:
        days = Surv_data1.loc[i]['days']
        event = Surv_data1.loc[i]['event']
        days = np.log2(days + 1)
        days_list.append(days)
        event_list.append(event)

    TCGA_gene["days"] = days_list
    TCGA_gene["event"] = event_list

    TCGA_gene = TCGA_gene[TCGA_gene['days'] != 0]

    TCGA_gene = TCGA_gene.iloc[:, 0:-1]

    Final_TCGA = TCGA_gene

    return Final_TCGA, Final_TCGA2
