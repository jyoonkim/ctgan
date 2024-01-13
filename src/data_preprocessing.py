import os
import pandas as pd
import pickle

def count_startswith(string1, list1):
    counter = 0
    for i in list1:
        counter += i.startswith(string1)
    return counter

def find_names(string1, list1):
    tmp = []
    for i in list1:
        if string1 in i:
            tmp.append(i)
    return tmp

def check_index(list1, list2, switch = 1):
    tmp = []
    if switch == 1:
        for i in list1:
            for j in range(len(list2)):
                if i + '-01' in list2[j]:
                    tmp.append(list2[j])
                    break
    elif switch == 2:
        for i in list1:
            for j in range(len(list2)):
                if 'reads_per_million_miRNA_mapped_' + i + '-01' in list2[j]:
                    tmp.append(list2[j])
                    break
    return tmp

def data_preprocess():
    TCGA_data_list = os.listdir('./../data/raw_ge_data/')

    cancer_list = []
    for k in range(len(TCGA_data_list)):
        cancer = TCGA_data_list[k][5:].split('.txt')[0]
        cancer_list.append(cancer)

    for k in range(len(cancer_list)):
        Surv_data = pd.read_csv('./../data/survival_data/event_days_' + cancer_list[k] + '.csv', sep=',', header=0, index_col=0)
        TCGA = pd.read_csv('./../data/raw_ge_data/'+TCGA_data_list[k], sep='\t', low_memory=False)

        patients = Surv_data.index.tolist()

        mrna_header = TCGA.index.to_list()


        tumor_sample = [] ### select only tumor samples
        for i in mrna_header:
            if i[12:14]=='-0':
                tumor_sample.append(i)
        tumor_sample_TCGA = TCGA.loc[tumor_sample]

        mrna_header = tumor_sample_TCGA.index.to_list()


        mrna_header_split = [] ### split only the part that can identify the sample from the TCGA barcode

        for i in range(len(mrna_header)):
            new_mrna_header = mrna_header[i][0:12]
            mrna_header_split.append(new_mrna_header)

        check_index = []  ###  samples present in TCGA data but not in survival data
        for i in mrna_header_split:
            count = count_startswith(i, patients)
            if count==0:
                check_index.append(i)

        find_mrna_header = []
        for i in range(len(check_index)):
            find_mrna_header = find_names(check_index[i], mrna_header)
            #print(find_mrna_header)
            tumor_sample_TCGA = tumor_sample_TCGA.drop(find_mrna_header)

        os.makedirs('./../data/preprocess_ge_data',exist_ok=True)

        with open('./../data/preprocess_ge_data/TCGA-' + cancer_list[k]+'_preprocess.txt', 'wb') as file:
            pickle.dump(tumor_sample_TCGA, file)

#data_preprocess()


