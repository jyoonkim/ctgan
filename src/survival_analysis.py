import numpy as np
import pandas as pd
import pickle
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from lifelines.utils import k_fold_cross_validation
import os
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from warnings import filterwarnings
from tqdm import tqdm

from evaluate_data_preprocess import evaluate_dataset_load

def c_index_dataframe(cancer_type = ['COAD'], significant_gene_num = 300, train_epoch =500, augmentation = 'yes', days_source = 'ctgan'):
    filterwarnings('ignore')
    #cancer_type = ['COAD']  ## If you want all cancer types -> ['BLCA', 'COAD','HNSC', 'KIRC', 'LGG','LIHC', 'LUAD', 'LUSC', 'OV', 'SKCM', 'STAD']

    for cancer in cancer_type:
        #print("cancer : %s"%cancer)

        Final_TCGA, fake_A, reconstruct_A, Final_TCGA2, fake_B, reconstruct_B = evaluate_dataset_load(cancer = cancer, DIR ='./../saved_models/', significant_gene_num  = significant_gene_num, train_epoch = train_epoch)

        f_name = './../c_index_evaluation/' + cancer + '/' + 'gene_' + str(significant_gene_num) + '_' + 'epoch_' + str(train_epoch) +'/'

        os.makedirs(f_name, exist_ok=True)

        if days_source=='ctgan':
            fake_TCGA = fake_A  ### (gene expression, days)
            TCGA = Final_TCGA   ### (gene expression, days, event)

            fake_TCGA['event'] = Final_TCGA2['event'].to_list()

            fake_TCGA['days'] = np.exp2(fake_TCGA['days']) - 1

            TCGA['days'] = np.exp2(TCGA['days']) - 1
            TCGA.reset_index(drop=True, inplace=True)

        else: ### days from BRCA
            fake_TCGA = fake_A  ### (gene expression, days)
            TCGA = Final_TCGA   ### (gene expression, days, event)

            #fake_TCGA = fake_TCGA.iloc[:, significant_gene_num]

            fake_TCGA['days'] = (np.exp2(Final_TCGA2['days']) - 1).to_list()   ### days from BRCA
            fake_TCGA['event'] = Final_TCGA2['event'].to_list()

            TCGA['days'] = np.exp2(TCGA['days']) - 1
            TCGA.reset_index(drop=True, inplace=True)

        ######################################## SuperPC ################################################
        pca_num_feature = 3

        all_train_data_c_index_list = []
        all_test_data_c_index_list = []

        for m in tqdm(range(0,30)):
            train_data, test_data = train_test_split(TCGA, test_size=0.2, shuffle=True, random_state=m*2000, stratify=TCGA['event'])

            if augmentation=='yes':
                total_train_data = pd.concat([train_data, fake_TCGA], axis=0)
                train_data = total_train_data

            train_data = train_data.reset_index(drop=True)
            test_data = test_data.reset_index(drop=True)

            train_data_gene = train_data.iloc[:, 0:-2]
            train_data_survival = train_data.iloc[:, -2:]
            test_data_gene = test_data.iloc[:, 0:-2]
            test_data_survival = test_data.iloc[:, -2:]


            column_list = [str(0), 'days', 'event']

            each_gene_score_superpc = []

            for i in range(train_data_gene.shape[1]):
                each_gene_event_one = pd.concat([train_data_gene.iloc[:,i], train_data_survival], axis=1)
                each_gene_event_one.columns = column_list
                each_gene_event_one = each_gene_event_one[each_gene_event_one['event']==1]
                each_gene_event_one = each_gene_event_one.reset_index(drop=True)

                each_gene_survival_data = pd.concat([train_data_gene.iloc[:,i], train_data_survival], axis=1)
                each_gene_survival_data.columns = column_list
                cph = CoxPHFitter()
                cph.fit(each_gene_survival_data, duration_col='days', event_col='event', step_size=0.1, show_progress=False)
                c_index_score = cph.score(each_gene_survival_data, scoring_method='concordance_index')
                each_gene_score_superpc.append(c_index_score)

            max_c_index = max(each_gene_score_superpc)
            min_c_index = min(each_gene_score_superpc)

            n_threshold = 20
            threshold_list = []

            difference = (max_c_index-min_c_index)/(n_threshold-1)

            for i in range(n_threshold):
                if i==(n_threshold-1):
                    threshold = max_c_index
                    threshold_list.append(threshold)
                else:
                    threshold = min_c_index + i*difference
                    threshold_list.append(threshold)

            k_fold_cross_val_c_index_list = []  ### len(k_fold_cross_val_c_index_list) ==len(n_threshold)

            for i in range(n_threshold):
                gene_higher_than_threshold = []  ### The collection of indices for genes whose scores are higher than the threshold for each threshold.
                for j in range(len(each_gene_score_superpc)):
                    if each_gene_score_superpc[j] > threshold_list[i]:
                        gene_higher_than_threshold.append(j)
                    if j==(len(each_gene_score_superpc)-1):
                        if len(gene_higher_than_threshold)<5:
                            pass
                            # print("There are less than FIVE")
                            # print("Threshold : %f"%threshold_list[i])
                        else:
                            subset_gene_df = train_data_gene.iloc[:,gene_higher_than_threshold]
                            my_pca = PCA(n_components=pca_num_feature, random_state=42).fit(subset_gene_df)

                            subset_gene_df = my_pca.transform(subset_gene_df)
                            subset_gene_df = pd.DataFrame(subset_gene_df)

                            subset_total_df = pd.concat([subset_gene_df, train_data_survival], axis=1)

                            cph = CoxPHFitter()
                            k_fold_cross_val_c_index = np.mean(k_fold_cross_validation(cph, subset_total_df, duration_col='days', event_col='event',fitter_kwargs={'step_size': 0.1}, seed=42, scoring_method="concordance_index"))
                            k_fold_cross_val_c_index_list.append(k_fold_cross_val_c_index)

            max_index = k_fold_cross_val_c_index_list.index(max(k_fold_cross_val_c_index_list))

            best_threshold = threshold_list[max_index]


            gene_higher_than_threshold = []
            for j in range(len(each_gene_score_superpc)):
                if each_gene_score_superpc[j] > best_threshold:
                    gene_higher_than_threshold.append(j)
                if j == (len(each_gene_score_superpc) - 1):
                    if len(gene_higher_than_threshold) < 5:
                        pass
                        # print("There are less than FIVE")
                        # print("Threshold : %f" % threshold_list[i])

                    else:
                        train_subset_gene_df = train_data_gene.iloc[:, gene_higher_than_threshold]
                        best_pca = PCA(n_components=pca_num_feature, random_state=42).fit(train_subset_gene_df)

                        train_subset_gene_df = best_pca.transform(train_subset_gene_df)
                        train_subset_gene_df = pd.DataFrame(train_subset_gene_df)

                        train_subset_total_df = pd.concat([train_subset_gene_df, train_data_survival], axis=1)

                        test_subset_gene_df = test_data_gene.iloc[:, gene_higher_than_threshold]
                        test_subset_gene_df = best_pca.transform(test_subset_gene_df)
                        test_subset_gene_df = pd.DataFrame(test_subset_gene_df)

                        test_subset_total_df = pd.concat([test_subset_gene_df, test_data_survival], axis=1)


                        cph = CoxPHFitter()
                        cph.fit(train_subset_total_df,duration_col='days', event_col='event',step_size=0.1, show_progress=False)
                        #cph.print_summary()
                        Best_train_data_c_index = cph.score(train_subset_total_df, scoring_method='concordance_index')
                        Best_test_data_c_index = cph.score(test_subset_total_df, scoring_method='concordance_index')

            all_train_data_c_index_list.append(Best_train_data_c_index)
            all_test_data_c_index_list.append(Best_test_data_c_index)

        # print("Mean train data c index for all random states : %f"%(sum(all_train_data_c_index_list)/len(all_train_data_c_index_list)))
        # print("Median train data c index for all rnadom states : %f"%statistics.median(all_train_data_c_index_list))
        # print("Mean test data c index for all random states : %f"%(sum(all_test_data_c_index_list)/len(all_test_data_c_index_list)))
        # print("Median test data c index for all rnadom states : %f"%statistics.median(all_test_data_c_index_list))

        c_index_dataframe = pd.DataFrame(data = list(zip(all_train_data_c_index_list,all_test_data_c_index_list)), columns=['Train','Test'])

        if augmentation == 'no':
            with open(f_name + '30_only_real_c_index_dataframe', "wb") as file:
                pickle.dump(c_index_dataframe, file)
        else:
            if days_source=='ctgan':
                with open(f_name + '30_real_fake_days_from_ctgan_c_index_dataframe',"wb") as file:
                    pickle.dump(c_index_dataframe, file)
            else:
                with open(f_name + '30_real_fake_days_from_BRCA_c_index_dataframe',"wb") as file:
                    pickle.dump(c_index_dataframe, file)

def boxplot_all_cancer_improvement_c_index(significant_gene_num = 300, train_epoch = 500):

    f_name = './../c_index_evaluation/boxplot'

    os.makedirs(f_name, exist_ok=True)

    cancer_type = ['BLCA', 'COAD', 'HNSC', 'KIRC', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'SKCM', 'STAD']

    number_models = 2
    model_name_list = ['days from ctGAN', 'days from real BRCA']

    cancer_type_list = np.concatenate((np.repeat('BLCA', 30 * number_models), np.repeat('COAD', 30 * number_models),
                                       np.repeat('HNSC', 30 * number_models), np.repeat('KIRC', 30 * number_models),
                                       np.repeat('LGG', 30 * number_models),
                                       np.repeat('LIHC', 30 * number_models), np.repeat('LUAD', 30 * number_models),
                                       np.repeat('LUSC', 30 * number_models),
                                       np.repeat('OV', 30 * number_models), np.repeat('SKCM', 30 * number_models),
                                       np.repeat('STAD', 30 * number_models)
                                       ))
    cancer_type_list = cancer_type_list.reshape(-1, 1)
    cancer_type_list = pd.DataFrame(cancer_type_list)

    model_type_list = np.concatenate((np.repeat(model_name_list[0], 30), np.repeat(model_name_list[1], 30)))
    for i in range(len(cancer_type) - 1):
        model_type_list = np.concatenate(
            (model_type_list, np.repeat(model_name_list[0], 30), np.repeat(model_name_list[1], 30)))
    model_type_list = model_type_list.reshape(-1, 1)
    model_type_list = pd.DataFrame(model_type_list)

    for cancer in cancer_type:

        with open('./../c_index_evaluation/' + cancer + '/' + 'gene_' + str(significant_gene_num) + '_' + 'epoch_' + str(train_epoch) +'/' + '30_real_fake_days_from_ctgan_c_index_dataframe', "rb") as file:
            first_model_real_fake_c_index_dataframe = pickle.load(file)

        with open('./../c_index_evaluation/' + cancer + '/' + 'gene_' + str(significant_gene_num) + '_' + 'epoch_' + str(train_epoch) +'/' + '30_real_fake_days_from_BRCA_c_index_dataframe', "rb") as file:
            second_model_real_fake_c_index_dataframe = pickle.load(file)

        with open('./../c_index_evaluation/' + cancer + '/' + 'gene_' + str(significant_gene_num) + '_' + 'epoch_' + str(train_epoch) +'/' + '30_only_real_c_index_dataframe', "rb") as file:
            only_real_c_index_dataframe_1 = pickle.load(file)

        all_c_index_dataframe = pd.concat([only_real_c_index_dataframe_1, first_model_real_fake_c_index_dataframe,
                                           second_model_real_fake_c_index_dataframe], axis=1)
        all_c_index_dataframe.columns = ['Original Train', 'Original Test', 'First Train', 'First Test',
                                         'Second Train', 'Second Test']
        new_col = ['Original Train', 'First Train', 'Second Train'] + ['Original Test', 'First Test', 'Second Test']
        all_c_index_dataframe = all_c_index_dataframe[new_col]

        first_model_test_improvement = all_c_index_dataframe.iloc[:, 4] - all_c_index_dataframe.iloc[:, 3]
        second_model_test_improvement = all_c_index_dataframe.iloc[:, 5] - all_c_index_dataframe.iloc[:, 3]

        if cancer == 'BLCA':
            first_model_test_improvement_BLCA = first_model_test_improvement
            second_model_test_improvement_BLCA = second_model_test_improvement
        if cancer == 'COAD':
            first_model_test_improvement_COAD = first_model_test_improvement
            second_model_test_improvement_COAD = second_model_test_improvement
        if cancer == 'HNSC':
            first_model_test_improvement_HNSC = first_model_test_improvement
            second_model_test_improvement_HNSC = second_model_test_improvement
        if cancer == 'KIRC':
            first_model_test_improvement_KIRC = first_model_test_improvement
            second_model_test_improvement_KIRC = second_model_test_improvement
        if cancer == 'LGG':
            first_model_test_improvement_LGG = first_model_test_improvement
            second_model_test_improvement_LGG = second_model_test_improvement
        if cancer == 'LIHC':
            first_model_test_improvement_LIHC = first_model_test_improvement
            second_model_test_improvement_LIHC = second_model_test_improvement
        if cancer == 'LUAD':
            first_model_test_improvement_LUAD = first_model_test_improvement
            second_model_test_improvement_LUAD = second_model_test_improvement
        if cancer == 'LUSC':
            first_model_test_improvement_LUSC = first_model_test_improvement
            second_model_test_improvement_LUSC = second_model_test_improvement
        if cancer == 'OV':
            first_model_test_improvement_OV = first_model_test_improvement
            second_model_test_improvement_OV = second_model_test_improvement
        if cancer == 'SKCM':
            first_model_test_improvement_SKCM = first_model_test_improvement
            second_model_test_improvement_SKCM = second_model_test_improvement
        if cancer == 'STAD':
            first_model_test_improvement_STAD = first_model_test_improvement
            second_model_test_improvement_STAD = second_model_test_improvement

    improvement_c_index_dataframe = pd.concat(
        [first_model_test_improvement_BLCA, second_model_test_improvement_BLCA,
         first_model_test_improvement_COAD, second_model_test_improvement_COAD,
         first_model_test_improvement_HNSC, second_model_test_improvement_HNSC,
         first_model_test_improvement_KIRC, second_model_test_improvement_KIRC,
         first_model_test_improvement_LGG, second_model_test_improvement_LGG,
         first_model_test_improvement_LIHC, second_model_test_improvement_LIHC,
         first_model_test_improvement_LUAD, second_model_test_improvement_LUAD,
         first_model_test_improvement_LUSC, second_model_test_improvement_LUSC,
         first_model_test_improvement_OV, second_model_test_improvement_OV, first_model_test_improvement_SKCM,
         second_model_test_improvement_SKCM, first_model_test_improvement_STAD,
         second_model_test_improvement_STAD], axis=0)
    improvement_c_index_dataframe = np.asarray(improvement_c_index_dataframe).reshape(-1, 1)
    improvement_c_index_dataframe = pd.DataFrame(improvement_c_index_dataframe)
    final_improvement_c_index_df = pd.concat([improvement_c_index_dataframe, cancer_type_list], axis=1)
    final_improvement_c_index_df = pd.concat([final_improvement_c_index_df, model_type_list], axis=1)
    final_improvement_c_index_df.columns = ['improved_c_index_value', 'cancer', 'model']

    plt.figure(figsize=(10, 4))
    sns.set_style('whitegrid')
    sns.set_style('ticks')

    my_pal = {model_name_list[0]: "salmon", model_name_list[1]: "khaki"}

    ax = sns.boxplot(width=0.4, showfliers=False, data=final_improvement_c_index_df, palette=my_pal,
                     medianprops=dict(color='k', linewidth=1.2), capprops=dict(color='k', linewidth=0.8),
                     whiskerprops=dict(color='k', linewidth=0.8), fliersize=1.0, hue='model', x='cancer',
                     y='improved_c_index_value')

    for i in range(len(cancer_type) * number_models):
        mybox = ax.artists[i]
        mybox.set_edgecolor('black')
        mybox.set_linewidth(0.8)
        mybox.set_alpha(0.8)

    plt.ylabel("C-Index values", fontsize=11)
    plt.xlabel("", fontsize=3)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=7.5)
    plt.axhline(y=0.0, color='red', linewidth=1.0, linestyle='--')
    plt.legend(fontsize=10, loc='lower right')
    # plt.show()

    os.makedirs(f_name + '/gene_' + str(significant_gene_num) + '_' + 'epoch_' + str(train_epoch) + '/' , exist_ok=True)
    plt.savefig(f_name + '/gene_' + str(significant_gene_num) + '_' + 'epoch_' + str(train_epoch) + '/'  + '(11 cancers)ctGAN_improvement_c_index_value_boxplot' + '.tiff', dpi=350)

def boxplot_one_cancer_improvement_c_index(cancer = 'COAD',significant_gene_num = 300, train_epoch = 500):

    f_name = './../c_index_evaluation/boxplot'

    os.makedirs(f_name, exist_ok=True)

    number_models = 2
    model_name_list = ['days from ctGAN', 'days from real BRCA']

    cancer_type_list = np.repeat(cancer, 30*number_models)
    cancer_type_list = cancer_type_list.reshape(-1, 1)
    cancer_type_list = pd.DataFrame(cancer_type_list)

    model_type_list = np.concatenate((np.repeat(model_name_list[0], 30), np.repeat(model_name_list[1], 30)))
    model_type_list = model_type_list.reshape(-1, 1)
    model_type_list = pd.DataFrame(model_type_list)

    with open('./../c_index_evaluation/' + cancer + '/' + 'gene_' + str(significant_gene_num) + '_' + 'epoch_' + str(train_epoch) +'/' + '30_real_fake_days_from_ctgan_c_index_dataframe', "rb") as file:
        first_model_real_fake_c_index_dataframe = pickle.load(file)

    with open('./../c_index_evaluation/' + cancer + '/' + 'gene_' + str(significant_gene_num) + '_' + 'epoch_' + str(train_epoch) +'/' + '30_real_fake_days_from_BRCA_c_index_dataframe', "rb") as file:
        second_model_real_fake_c_index_dataframe = pickle.load(file)

    with open('./../c_index_evaluation/' + cancer + '/' + 'gene_' + str(significant_gene_num) + '_' + 'epoch_' + str(train_epoch) +'/' + '30_only_real_c_index_dataframe', "rb") as file:
        only_real_c_index_dataframe_1 = pickle.load(file)

    all_c_index_dataframe = pd.concat([only_real_c_index_dataframe_1, first_model_real_fake_c_index_dataframe,
                                       second_model_real_fake_c_index_dataframe], axis=1)
    all_c_index_dataframe.columns = ['Original Train', 'Original Test', 'First Train', 'First Test',
                                     'Second Train', 'Second Test']
    new_col = ['Original Train', 'First Train', 'Second Train'] + ['Original Test', 'First Test', 'Second Test']
    all_c_index_dataframe = all_c_index_dataframe[new_col]

    first_model_test_improvement = all_c_index_dataframe.iloc[:, 4] - all_c_index_dataframe.iloc[:, 3]
    second_model_test_improvement = all_c_index_dataframe.iloc[:, 5] - all_c_index_dataframe.iloc[:, 3]

    improvement_c_index_dataframe = pd.concat([first_model_test_improvement, second_model_test_improvement], axis=0)
    improvement_c_index_dataframe = np.asarray(improvement_c_index_dataframe).reshape(-1, 1)
    improvement_c_index_dataframe = pd.DataFrame(improvement_c_index_dataframe)
    final_improvement_c_index_df = pd.concat([improvement_c_index_dataframe, cancer_type_list], axis=1)
    final_improvement_c_index_df = pd.concat([final_improvement_c_index_df, model_type_list], axis=1)
    final_improvement_c_index_df.columns = ['improved_c_index_value', 'cancer', 'model']

    plt.figure(figsize=(5, 5))
    sns.set_style('whitegrid')
    sns.set_style('ticks')

    my_pal = {model_name_list[0]: "salmon", model_name_list[1]: "khaki"}

    ax = sns.boxplot(width=0.15, showfliers=False, data=final_improvement_c_index_df, palette=my_pal,
                     medianprops=dict(color='k', linewidth=1.2), capprops=dict(color='k', linewidth=0.8),
                     whiskerprops=dict(color='k', linewidth=0.8), fliersize=1.0, hue='model', x='cancer',
                     y='improved_c_index_value')


    for i in range(number_models):
        mybox = ax.artists[i]
        mybox.set_edgecolor('black')
        mybox.set_linewidth(0.8)
        mybox.set_alpha(0.8)

    plt.ylabel("C-Index values", fontsize=11)
    plt.xlabel("", fontsize=3)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=7.5)
    plt.axhline(y=0.0, color='red', linewidth=1.0, linestyle='--')
    plt.legend(fontsize=7, loc='lower right')
    # plt.show()

    os.makedirs(f_name + '/gene_' + str(significant_gene_num) + '_' + 'epoch_' + str(train_epoch) + '/' , exist_ok=True)
    plt.savefig(f_name + '/gene_' + str(significant_gene_num) + '_' + 'epoch_' + str(train_epoch) + '/' + '(' + cancer + ')'  + 'ctGAN_improvement_c_index_value_boxplot' + '.tiff', dpi=350)
