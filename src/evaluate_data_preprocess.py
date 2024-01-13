import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

def evaluate_dataset_load(cancer = 'COAD', DIR ='./../saved_models/', significant_gene_num  = 300, train_epoch = 500):
    # cancer_type = ['COAD'] ## If you want all cancer types -> ['BLCA', 'COAD','HNSC', 'KIRC', 'LGG','LIHC', 'LUAD', 'LUSC', 'OV', 'SKCM', 'STAD']

    Surv_data = pd.read_csv('./../data/survival_data/event_days_'+cancer+'.csv', sep=',', header=0, index_col=0)

    with open('./../data/preprocess_ge_data/TCGA-' + cancer + '_preprocess.txt','rb') as file:
        TCGA = pickle.load(file)

    with open('./../data/significant_gene/' + cancer + '/'+str(significant_gene_num)+'_Survival_related_significant_gene_list','rb') as f:
        significant_genes = pickle.load(f)

    TCGA = TCGA.iloc[:,significant_genes]

    TCGA = np.log2(TCGA + 1)
    ############################################
    patients = Surv_data.index.tolist()

    mrna_header = TCGA.index.to_list()

    days_list = []
    event_list = []

    split_mrna_header_list = []

    for i in mrna_header:
        split_mrna_header = i[0:12]
        split_mrna_header_list.append(split_mrna_header)

    for i in split_mrna_header_list:
        days = Surv_data.loc[i]['days']
        event = Surv_data.loc[i]['event']
        days= np.log2(days+1)
        days_list.append(days)
        event_list.append(event)

    TCGA["days"] = days_list
    TCGA["event"] = event_list

    TCGA = TCGA[TCGA['days'] !=0]

    TCGA_gene = TCGA.iloc[:, 0:-1]  ## (gene expression,days)

    A_train_data, A_test_data = train_test_split(TCGA_gene, test_size=0.1, shuffle=True, random_state=0)
    original_A_train = A_train_data.copy()

    TCGA_column_list = TCGA_gene.columns.to_list()[0:significant_gene_num+1]

    ##########################################
    with open(DIR+ cancer + '/gene_' + str(significant_gene_num) + '/All_fake_A_' + str(train_epoch), 'rb') as file:
        fake_TCGA = pickle.load(file)
    with open(DIR + cancer + '/gene_' + str(significant_gene_num) + '/All_reconstruction_A_' + str(train_epoch), 'rb') as file:
        reconstruct_TCGA = pickle.load(file)

    fake_TCGA = fake_TCGA if isinstance(fake_TCGA, np.ndarray) else fake_TCGA.cpu().detach().numpy()
    fake_TCGA = pd.DataFrame(fake_TCGA)

    reconstruct_TCGA = reconstruct_TCGA if isinstance(reconstruct_TCGA, np.ndarray) else reconstruct_TCGA.cpu().detach().numpy()
    reconstruct_TCGA = pd.DataFrame(reconstruct_TCGA)

    original_A_train.reset_index(drop=True, inplace=True)

    column_list = []
    for m in range(significant_gene_num+1):
        column_list.append(m)
    original_A_train.columns = column_list

    fake_TCGA = (fake_TCGA * (original_A_train.std() + 1.2e-17)) + original_A_train.mean()
    reconstruct_TCGA = (reconstruct_TCGA * (original_A_train.std() + 1.2e-17)) + original_A_train.mean()

    fake_TCGA.columns = TCGA_column_list
    reconstruct_TCGA.columns = TCGA_column_list

    fake_A = fake_TCGA ## (gene expression, days) all log2
    reconstruct_A = reconstruct_TCGA ## (gene expression, days) all log2
    Final_TCGA = TCGA ## (gene expression, days, event) all log2 except event

    ########################################################################################################################
    Surv_data2 = pd.read_csv('./../data/survival_data/event_days_' + 'BRCA' + '.csv',sep=',', header=0, index_col=0)

    with open('./../data/preprocess_ge_data/TCGA-BRCA_preprocess.txt','rb') as file:
        TCGA2 = pickle.load(file)

    with open('./../data/significant_gene/' + cancer + '/' + str(significant_gene_num) + '_Survival_related_significant_gene_list', 'rb') as f:
        significant_genes = pickle.load(f)

    TCGA2 = TCGA2.iloc[:, significant_genes]

    TCGA2 = np.log2(TCGA2 + 1)
    ##############################################################
    patients = Surv_data2.index.tolist()

    mrna_header = TCGA2.index.to_list()

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

    TCGA2["days"] = days_list
    TCGA2["event"] = event_list

    TCGA2 = TCGA2[TCGA2['days'] != 0]

    TCGA2_gene = TCGA2.iloc[:, 0:-1]

    B_train_data, B_test_data = train_test_split(TCGA2_gene, test_size=0.1, shuffle=True, random_state=0)
    original_B_train = B_train_data.copy()

    TCGA2_column_list = TCGA2_gene.columns.to_list()[0:significant_gene_num+1]

    ##########################################
    with open(DIR + cancer + '/gene_' + str(significant_gene_num) + '/All_fake_B_' + str(train_epoch), 'rb') as file:
        fake_TCGA2 = pickle.load(file)
    with open(DIR+ cancer + '/gene_' + str(significant_gene_num) + '/All_reconstruction_B_' + str(train_epoch), 'rb') as file:
        reconstruct_TCGA2 = pickle.load(file)

    fake_TCGA2 = fake_TCGA2 if isinstance(fake_TCGA2, np.ndarray) else fake_TCGA2.cpu().detach().numpy()
    fake_TCGA2 = pd.DataFrame(fake_TCGA2)

    reconstruct_TCGA2 = reconstruct_TCGA2 if isinstance(reconstruct_TCGA2, np.ndarray) else reconstruct_TCGA2.cpu().detach().numpy()
    reconstruct_TCGA2 = pd.DataFrame(reconstruct_TCGA2)

    original_B_train.reset_index(drop=True, inplace=True)

    column_list = []
    for m in range(significant_gene_num+1):
        column_list.append(m)
    original_B_train.columns = column_list

    fake_TCGA2 = (fake_TCGA2 * (original_B_train.std() + 1.2e-17)) + original_B_train.mean()
    reconstruct_TCGA2 = (reconstruct_TCGA2 * (original_B_train.std() + 1.2e-17)) + original_B_train.mean()

    fake_TCGA2.columns = TCGA2_column_list
    reconstruct_TCGA2.columns = TCGA2_column_list

    fake_B = fake_TCGA2 ## (gene expression, days) all log2
    reconstruct_B = reconstruct_TCGA2 ## (gene expression, days) all log2
    Final_TCGA2 = TCGA2 ## (gene expression, days, event) all log2 except event

    Final_TCGA.reset_index(drop=True, inplace=True)
    Final_TCGA2.reset_index(drop=True, inplace=True)

    return Final_TCGA, fake_A, reconstruct_A, Final_TCGA2, fake_B, reconstruct_B