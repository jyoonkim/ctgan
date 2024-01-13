import statistics
import numpy as np
import pandas as pd
import pickle
from lifelines import CoxPHFitter
import logging
import os
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
from tqdm import tqdm

def count_startswith(string1, list1):
    counter = 0
    for i in list1:
        counter += i.startswith(string1)
    return counter

def cox_significant_gene(cancer_type = ['COAD']):
    filterwarnings('ignore')
    #cancer_type = ['BRCA','BLCA', 'COAD','HNSC', 'KIRC', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'SKCM', 'STAD']

    os.makedirs('./../data/significant_gene', exist_ok=True)

    # logger = logging.getLogger()
    # logger.setLevel((logging.INFO))
    # formatter = logging.Formatter('%(asctime)s -%(message)s')
    #
    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(formatter)
    # logger.addHandler(stream_handler)
    #
    # file_handler = logging.FileHandler('./../data/significant_gene/all cancers cox result.log')
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    for cancer in cancer_type:
        #print("Cancer type is %s"%cancer)
        #logger.info("Cancer type is %s"%cancer)
        Surv_data = pd.read_csv('./../data/survival_data/event_days_' + cancer+ '.csv', sep=',', header=0, index_col=0)

        with open('./../data/preprocess_ge_data/TCGA-' + cancer+ '_preprocess.txt', 'rb') as file:
            TCGA = pickle.load(file)

        TCGA = np.log2(TCGA + 1)  #### ln에서 log2로 바꿈

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
            days_list.append(days)
            event_list.append(event)

        TCGA["days"] = days_list
        TCGA["event"] = event_list

        TCGA = TCGA[TCGA['days'] != 0]

        A_train_data, A_test_data = train_test_split(TCGA, test_size=0.1, shuffle=True, random_state=0)
        TCGA_gene = A_train_data.iloc[:, 0:-2]
        TCGA_survival = A_train_data.iloc[:, -2:]

        column_list = [str(0), 'days', 'event']

        each_gene_score_superpc = []
        survival_significant_gene_index_list = []

        for i in tqdm(range(TCGA_gene.shape[1])):
            each_gene_event_one = pd.concat([TCGA_gene.iloc[:, i], TCGA_survival], axis=1)
            each_gene_event_one.columns = column_list
            each_gene_event_one = each_gene_event_one[each_gene_event_one['event'] == 1]
            each_gene_event_one = each_gene_event_one.reset_index(drop=True)

            if each_gene_event_one.iloc[:, 0].std() < 1.2e-15:
                continue
            else:
                each_gene_survival_data = pd.concat([TCGA_gene.iloc[:, i], TCGA_survival], axis=1)
                each_gene_survival_data.columns = column_list
                cph = CoxPHFitter()
                cph.fit(each_gene_survival_data, duration_col='days', event_col='event', step_size=0.1, show_progress=False)
                c_index_score = cph.score(each_gene_survival_data, scoring_method='concordance_index')
                each_gene_score_superpc.append(c_index_score)
                survival_significant_gene_index_list.append(i)

        max_c_index = max(each_gene_score_superpc)
        min_c_index = min(each_gene_score_superpc)
        mean_c_index = sum(each_gene_score_superpc)/len(each_gene_score_superpc)
        median_c_index = statistics.median(each_gene_score_superpc)

        #print("max_c_index : %f"%max_c_index)
        #print("min_c_index : %f"%min_c_index)
        #print("min_c_index : %f"%mean_c_index)
        #print("median_c_index : %f"%median_c_index)

        # logger.info("max_c_index : %f"%max_c_index)
        # logger.info("min_c_index : %f"%min_c_index)
        # logger.info("min_c_index : %f"%mean_c_index)
        # logger.info("median_c_index : %f"%median_c_index)

        os.makedirs('./../data/significant_gene/' + cancer, exist_ok=True)

        with open('./../data/significant_gene/' + cancer + '/' + 'Survival_related_significant_gene_list','wb') as file:
            pickle.dump(survival_significant_gene_index_list, file)

        with open('./../data/significant_gene/' + cancer + '/' + 'Survival_related_significant_each_gene_score','wb') as file:
            pickle.dump(each_gene_score_superpc, file)


def sorted_significant_gene(cancer_type = ['COAD']):
    #cancer_type = ['BRCA','BLCA', 'COAD','HNSC', 'KIRC', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'SKCM', 'STAD']

    for cancer in cancer_type:
        #print("cancer : %s"%cancer)
        with open('./../data/significant_gene/' + cancer + '/' + 'Survival_related_significant_gene_list','rb') as file:
            survival_significant_gene_index_list = pickle.load(file)

        with open('./../data/significant_gene/' + cancer + '/' + 'Survival_related_significant_each_gene_score','rb') as file:
            each_gene_score_superpc = pickle.load(file)

        gene_score_sorted_index = np.array(each_gene_score_superpc).argsort()[::-1]  #### return the list index values of the sorted in descending order
        gene_score_sorted_index = gene_score_sorted_index.tolist()

        gene_index_sorted_array = np.asarray(survival_significant_gene_index_list)[gene_score_sorted_index]
        gene_index_sorted_list = gene_index_sorted_array.tolist()

        number_select_gene_list = [100,200,300,500,800,1000]
        for number_select_gene in number_select_gene_list:
            final_survival_significant_gene_index_list = gene_index_sorted_list[0:number_select_gene]
            os.makedirs('./../data/significant_gene/'+cancer, exist_ok=True)
            with open('./../data/significant_gene/'+cancer+'/'+str(number_select_gene) + '_Survival_related_significant_gene_list', 'wb') as file:
                pickle.dump(final_survival_significant_gene_index_list, file)








