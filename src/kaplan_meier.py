import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from lifelines import CoxPHFitter
from sklearn.decomposition import PCA
from lifelines.utils import k_fold_cross_validation
import os
import statistics
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import pickle
import math
import seaborn as sns
from warnings import filterwarnings
from tqdm import tqdm

from evaluate_data_preprocess import evaluate_dataset_load

#########################
def log_rank_p_value(cancer_type = ['COAD'], significant_gene_num = 300, train_epoch = 500, augmentation = 'yes'):
    filterwarnings('ignore')
    #cancer_type = ['COAD']  ## If you want all cancer types -> ['BLCA', 'COAD','HNSC', 'KIRC', 'LGG','LIHC', 'LUAD', 'LUSC', 'OV', 'SKCM', 'STAD']

    for cancer in cancer_type:
        #print("Cancer : %s"%cancer)

        Final_TCGA, fake_A, reconstruct_A, Final_TCGA2, fake_B, reconstruct_B = evaluate_dataset_load(cancer = cancer, DIR ='./../saved_models/', significant_gene_num  = significant_gene_num, train_epoch = train_epoch)

        f_name = './../kaplan_meier_evaluation/' + cancer + '/' + 'gene_' + str(significant_gene_num) + '_' + 'epoch_' + str(train_epoch)

        os.makedirs(f_name, exist_ok=True)

        fake_TCGA = fake_A  ### (gene expression, days)
        TCGA = Final_TCGA   ### (gene expression, days, event)

        fake_TCGA['event'] = Final_TCGA2['event'].to_list()

        fake_TCGA['days'] = np.exp2(fake_TCGA['days']) - 1

        TCGA['days'] = np.exp2(TCGA['days']) - 1
        TCGA.reset_index(drop=True, inplace=True)

        ######################################## SuperPC ################################################
        pca_num_feature = 3

        all_randomstate_log_rank_p_value_list = []

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

                        ########################################################################################
                        train_partial_hazard_list = cph.predict_partial_hazard(train_subset_total_df).to_list()
                        train_median_hazard_value = statistics.median(train_partial_hazard_list)
                        partial_hazard_list = cph.predict_partial_hazard(test_subset_total_df).to_list()
                        high_risk_group_list = [index for index, value in enumerate(partial_hazard_list) if
                                                value >= train_median_hazard_value]
                        low_risk_group_list = [index for index, value in enumerate(partial_hazard_list) if
                                               value < train_median_hazard_value]

                        high_risk_group_df = test_subset_total_df.iloc[high_risk_group_list]
                        low_risk_group_df = test_subset_total_df.iloc[low_risk_group_list]

                        high_risk_kmf = KaplanMeierFitter()
                        high_risk_kmf.fit(high_risk_group_df["days"], high_risk_group_df["event"], label='High risk group')

                        low_risk_kmf = KaplanMeierFitter()
                        low_risk_kmf.fit(low_risk_group_df["days"], low_risk_group_df["event"], label='Low risk group')

                        plt.figure(figsize=(10, 8))
                        plot = high_risk_kmf.plot_survival_function(show_censors=True, ci_show=False, linewidth=2.5,
                                                                    style='-',
                                                                    c='red')
                        plot = low_risk_kmf.plot_survival_function(show_censors=True, ci_show=False, linewidth=2.5,
                                                                   style='--',
                                                                   c='green')

                        log_rank_p_value = logrank_test(high_risk_group_df['days'], low_risk_group_df['days'],
                                                        high_risk_group_df['event'], low_risk_group_df['event']).p_value

                        all_randomstate_log_rank_p_value_list.append(log_rank_p_value)

                        # plot.legend(loc='upper right', fontsize=8)
                        #
                        # plot.set_xlabel('Time (days)', fontsize=14)
                        # plot.set_ylabel('Survival probability', fontsize=14)
                        #
                        # if augmentation == 'no':
                        #     os.makedirs(f_name + '/no_augmentation/', exist_ok=True)
                        #     plot.figure.savefig(f_name + '/no_augmentation/' + 'R' + str(m) + '_' + 'kaplan meier' + '.tiff' , dpi=350)
                        # else:
                        #    os.makedirs(f_name + '/yes_augmentation/', exist_ok=True)
                        #     plot.figure.savefig(f_name + '/yes_augmentation/' + 'R' + str(m) + '_' + 'kaplan meier' + '.tiff' , dpi=350)

        if augmentation == 'no':
            os.makedirs(f_name + '/no_augmentation/', exist_ok=True)
            with open(f_name + '/no_augmentation/' + '30_logrank_pvalue_dataframe', "wb") as file:
                pickle.dump(all_randomstate_log_rank_p_value_list, file)
        else:
            os.makedirs(f_name + '/yes_augmentation/', exist_ok=True)
            with open(f_name + '/yes_augmentation/' + '30_logrank_pvalue_dataframe', "wb") as file:
                pickle.dump(all_randomstate_log_rank_p_value_list, file)

def kaplan_meier_plot_complete_form_year(cancer_type = ['COAD'], significant_gene_num = 300, train_epoch = 500, augmentation = 'yes'):
    filterwarnings('ignore')

    for cancer in cancer_type:
        #print("Cancer : %s"%cancer)
        f_name = './../kaplan_meier_evaluation/' + cancer + '/' + 'gene_' + str(significant_gene_num) + '_' + 'epoch_' + str(train_epoch)

        os.makedirs(f_name, exist_ok=True)

        Surv_data = pd.read_csv('./../data/survival_data/event_days_' + cancer + '.csv',sep=',', header=0, index_col=0)

        complete_form_year_df = pd.read_csv('./../data/clinical_info_data/complete_form_year_' + cancer + '.csv',
                                            sep=',', header=0, index_col=0)

        with open('./../data/preprocess_ge_data/TCGA-' + cancer + '_preprocess.txt', 'rb') as file:
            TCGA = pickle.load(file)

        with open('./../data/significant_gene/' + cancer + '/' + str(significant_gene_num) + '_Survival_related_significant_gene_list', 'rb') as f:
            significant_genes = pickle.load(f)

        TCGA = TCGA.iloc[:, significant_genes]

        TCGA = np.log2(TCGA + 1)
        #####################################################
        patients = Surv_data.index.tolist()

        mrna_header = TCGA.index.to_list()

        days_list = []
        event_list = []
        complete_form_year_list = []

        split_mrna_header_list = []

        for i in mrna_header:
            split_mrna_header = i[0:12]
            split_mrna_header_list.append(split_mrna_header)

        for i in split_mrna_header_list:
            days = Surv_data.loc[i]['days']
            event = Surv_data.loc[i]['event']
            complete_year = complete_form_year_df.loc[i]['complete_year']

            days = np.log2(days + 1)
            days_list.append(days)
            event_list.append(event)
            complete_form_year_list.append(complete_year)

        TCGA["days"] = days_list
        TCGA["event"] = event_list
        TCGA["complete year"] = complete_form_year_list

        TCGA = TCGA[TCGA['days'] != 0]

        TCGA_gene = TCGA.iloc[:, 0:-2]

        A_train_data, A_test_data = train_test_split(TCGA_gene, test_size=0.1, shuffle=True, random_state=0)
        original_A_train = A_train_data.copy()

        TCGA_column_list = TCGA_gene.columns.to_list()[0:significant_gene_num + 1]

        ##########################################
        with open('./../saved_models/' + cancer + '/gene_' + str(significant_gene_num) + '/All_fake_A_' + str(train_epoch),
                  'rb') as file:
            fake_TCGA = pickle.load(file)
        with open('./../saved_models/' + cancer + '/gene_' + str(significant_gene_num) + '/All_reconstruction_A_' + str(train_epoch),
                  'rb') as file:
            reconstruct_TCGA = pickle.load(file)

        fake_TCGA = fake_TCGA if isinstance(fake_TCGA, np.ndarray) else fake_TCGA.cpu().detach().numpy()
        fake_TCGA = pd.DataFrame(fake_TCGA)

        reconstruct_TCGA = reconstruct_TCGA if isinstance(reconstruct_TCGA,
                                                          np.ndarray) else reconstruct_TCGA.cpu().detach().numpy()
        reconstruct_TCGA = pd.DataFrame(reconstruct_TCGA)

        original_A_train.reset_index(drop=True, inplace=True)

        column_list = []
        for m in range(significant_gene_num + 1):
            column_list.append(m)
        original_A_train.columns = column_list

        fake_TCGA = (fake_TCGA * (original_A_train.std() + 1.2e-17)) + original_A_train.mean()  ########### 여기 추가
        reconstruct_TCGA = (reconstruct_TCGA * (original_A_train.std() + 1.2e-17)) + original_A_train.mean()

        fake_TCGA.columns = TCGA_column_list
        reconstruct_TCGA.columns = TCGA_column_list

        fake_A = fake_TCGA  ###(gene expression, days)
        reconstruct_A = reconstruct_TCGA  #### (gene expression, days)
        Final_TCGA = TCGA  ### (gene expression, days, event, initial year)
        Final_TCGA.reset_index(drop=True, inplace=True)
        ##########################################################################
        _, _, _, Final_TCGA2, _, _ = evaluate_dataset_load(cancer=cancer, DIR='./../saved_models/',
                                                           significant_gene_num=significant_gene_num,
                                                           train_epoch=train_epoch)

        fake_TCGA = fake_A
        TCGA = Final_TCGA

        fake_TCGA['event'] = Final_TCGA2['event'].to_list()

        fake_TCGA['days'] = np.exp2(fake_TCGA['days']) - 1  ### (gene expression, days)

        TCGA['days'] = np.exp2(TCGA['days']) - 1
        TCGA.reset_index(drop=True, inplace=True)  ### (gene expression, days, event)

        ######################################## SuperPC ################################################
        pca_num_feature = 3

        train_idx = [index for index, value in enumerate(TCGA['complete year']) if
                     value < np.percentile(TCGA['complete year'], q=[70])]
        test_idx = [index for index, value in enumerate(TCGA['complete year']) if
                    value >= np.percentile(TCGA['complete year'], q=[70])]

        train_data = TCGA.iloc[train_idx]
        test_data = TCGA.iloc[test_idx]

        train_data = train_data.iloc[:, 0:-1]
        test_data = test_data.iloc[:, 0:-1]
        # train_data, test_data = train_test_split(TCGA, test_size=0.2, shuffle=True, random_state=m*2000, stratify=TCGA['event'])

        if augmentation == 'yes':
            total_train_data = pd.concat([train_data, fake_TCGA], axis=0)  ### 이 상태는 train에다가 augment 다 시킨 것
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
            each_gene_event_one = pd.concat([train_data_gene.iloc[:, i], train_data_survival], axis=1)
            each_gene_event_one.columns = column_list
            each_gene_event_one = each_gene_event_one[each_gene_event_one['event'] == 1]
            each_gene_event_one = each_gene_event_one.reset_index(drop=True)

            each_gene_survival_data = pd.concat([train_data_gene.iloc[:, i], train_data_survival], axis=1)
            each_gene_survival_data.columns = column_list
            cph = CoxPHFitter()
            cph.fit(each_gene_survival_data, duration_col='days', event_col='event', step_size=0.1, show_progress=False)
            c_index_score = cph.score(each_gene_survival_data, scoring_method='concordance_index')
            each_gene_score_superpc.append(c_index_score)

        max_c_index = max(each_gene_score_superpc)
        min_c_index = min(each_gene_score_superpc)

        n_threshold = 20
        threshold_list = []

        difference = (max_c_index - min_c_index) / (n_threshold - 1)

        for i in range(n_threshold):
            if i == (n_threshold - 1):
                threshold = max_c_index
                threshold_list.append(threshold)
            else:
                threshold = min_c_index + i * difference
                threshold_list.append(threshold)

        k_fold_cross_val_c_index_list = []  ### len(k_fold_cross_val_c_index_list) ==len(n_threshold)

        for i in range(n_threshold):
            gene_higher_than_threshold = []  ### The collection of indices for genes whose scores are higher than the threshold for each threshold.
            for j in range(len(each_gene_score_superpc)):
                if each_gene_score_superpc[j] > threshold_list[i]:
                    gene_higher_than_threshold.append(j)
                if j == (len(each_gene_score_superpc) - 1):
                    if len(gene_higher_than_threshold) < 5:
                        pass
                        # print("There are less than FIVE")
                        # print("Threshold : %f" % threshold_list[i])

                    else:
                        subset_gene_df = train_data_gene.iloc[:, gene_higher_than_threshold]
                        my_pca = PCA(n_components=pca_num_feature, random_state=42).fit(subset_gene_df)

                        subset_gene_df = my_pca.transform(subset_gene_df)
                        subset_gene_df = pd.DataFrame(subset_gene_df)

                        subset_total_df = pd.concat([subset_gene_df, train_data_survival], axis=1)

                        cph = CoxPHFitter()
                        k_fold_cross_val_c_index = np.mean(
                            k_fold_cross_validation(cph, subset_total_df, duration_col='days', event_col='event',
                                                    fitter_kwargs={'step_size': 0.1}, seed=42,
                                                    scoring_method="concordance_index"))
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
                    cph.fit(train_subset_total_df, duration_col='days', event_col='event', step_size=0.1,show_progress=False)
                    #############################################################################
                    train_partial_hazard_list = cph.predict_partial_hazard(train_subset_total_df).to_list()
                    train_median_hazard_value = statistics.median(train_partial_hazard_list)
                    partial_hazard_list = cph.predict_partial_hazard(test_subset_total_df).to_list()
                    high_risk_group_list = [index for index, value in enumerate(partial_hazard_list) if
                                            value >= train_median_hazard_value]
                    low_risk_group_list = [index for index, value in enumerate(partial_hazard_list) if
                                           value < train_median_hazard_value]

                    high_risk_group_df = test_subset_total_df.iloc[high_risk_group_list]
                    low_risk_group_df = test_subset_total_df.iloc[low_risk_group_list]

                    high_risk_kmf = KaplanMeierFitter()
                    high_risk_kmf.fit(high_risk_group_df["days"], high_risk_group_df["event"], label='High risk group')

                    low_risk_kmf = KaplanMeierFitter()
                    low_risk_kmf.fit(low_risk_group_df["days"], low_risk_group_df["event"], label='Low risk group')

                    plt.figure(figsize=(10, 8))
                    plot = high_risk_kmf.plot_survival_function(show_censors=True, ci_show=False, linewidth=2.5,
                                                                style='-',
                                                                c='red')
                    plot = low_risk_kmf.plot_survival_function(show_censors=True, ci_show=False, linewidth=2.5,
                                                               style='--',
                                                               c='green')

                    log_rank_p_value = logrank_test(high_risk_group_df['days'], low_risk_group_df['days'],
                                                    high_risk_group_df['event'], low_risk_group_df['event']).p_value

                    print("log rank p value: %.3f" % log_rank_p_value)

                    plot.legend(loc='upper right', fontsize=17)

                    plot.set_xlabel('Time (days)', fontsize=25)
                    plot.set_ylabel('Survival probability', fontsize=25)
                    plot.tick_params(axis='x', labelsize=18)
                    plot.tick_params(axis='y', labelsize=18)

                    if augmentation == 'no':
                        os.makedirs(f_name + '/no_augmentation/', exist_ok=True)
                        plot.figure.savefig(f_name + '/no_augmentation/' + 'pathology year kaplan meier' + '.tiff', dpi=350)
                    else:
                        os.makedirs(f_name + '/yes_augmentation/', exist_ok=True)
                        plot.figure.savefig(f_name + '/yes_augmentation/' + 'pathology year kaplan meier' + '.tiff', dpi=350)


def boxplot_all_cancer_improvement_log_rank_p_value(significant_gene_num = 300, train_epoch = 500): ### plot for 11 cancers, only use when augmentation = 'yes'

    f_name = './../kaplan_meier_evaluation/boxplot'

    os.makedirs(f_name, exist_ok=True)

    cancer_type = ['BLCA',  'COAD', 'HNSC', 'KIRC', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'SKCM', 'STAD']

    number_models = 1
    model_name_list = ['ctGAN']

    cancer_type_list = np.concatenate((np.repeat('BLCA', 30 * number_models),  np.repeat('COAD', 30 * number_models),
                                       np.repeat('HNSC', 30 * number_models), np.repeat('KIRC', 30 * number_models), np.repeat('LGG', 30 * number_models),
                                       np.repeat('LIHC', 30 * number_models), np.repeat('LUAD', 30 * number_models), np.repeat('LUSC', 30 * number_models),
                                       np.repeat('OV', 30 * number_models), np.repeat('SKCM', 30 * number_models), np.repeat('STAD', 30 * number_models),
                                       ))
    cancer_type_list = cancer_type_list.reshape(-1, 1)
    cancer_type_list = pd.DataFrame(cancer_type_list)

    model_type_list = np.repeat(model_name_list[0], 30)

    for i in range(len(cancer_type)-1):
        model_type_list = np.concatenate((model_type_list, np.repeat(model_name_list[0], 30)))
    model_type_list = model_type_list.reshape(-1, 1)
    model_type_list = pd.DataFrame(model_type_list)

    for cancer in cancer_type:

        with open('./../kaplan_meier_evaluation/' + cancer + '/' + 'gene_' + str(significant_gene_num) + '_' + 'epoch_' + str(train_epoch) + '/no_augmentation/' + '30_logrank_pvalue_dataframe', "rb") as file:
            only_real_logrank_pvalue_list = pickle.load(file)

        with open('./../kaplan_meier_evaluation/' + cancer + '/' + 'gene_' + str(significant_gene_num) + '_' + 'epoch_' + str(train_epoch) + '/yes_augmentation/' + '30_logrank_pvalue_dataframe',"rb") as file:
            first_model_real_fake_logrank_pvalue_list = pickle.load(file)


        new_only_real_logrank_pvalue_list = []
        for i in range(30):
            value = (math.log10(only_real_logrank_pvalue_list[i]))*-1
            new_only_real_logrank_pvalue_list.append(value)

        new_first_model_real_fake_logrank_pvalue_list = []
        for i in range(30):
            value = (math.log10(first_model_real_fake_logrank_pvalue_list[i]))*-1
            new_first_model_real_fake_logrank_pvalue_list.append(value)


        new_only_real_logrank_pvalue_list_df = pd.DataFrame(new_only_real_logrank_pvalue_list)
        new_first_model_real_fake_logrank_pvalue_df = pd.DataFrame(new_first_model_real_fake_logrank_pvalue_list)

        all_c_index_dataframe = pd.concat([new_only_real_logrank_pvalue_list_df,  new_first_model_real_fake_logrank_pvalue_df], axis=1)
        all_c_index_dataframe.columns = ['none', 'ctGAN']

        if cancer == 'BLCA':
            first_model_all_c_index_df_BLCA = all_c_index_dataframe['ctGAN'] - all_c_index_dataframe['none']
        if cancer == 'COAD':
            first_model_all_c_index_df_COAD = all_c_index_dataframe['ctGAN']- all_c_index_dataframe['none']
        if cancer == 'HNSC':
            first_model_all_c_index_df_HNSC = all_c_index_dataframe['ctGAN']- all_c_index_dataframe['none']
        if cancer == 'KIRC':
            first_model_all_c_index_df_KIRC = all_c_index_dataframe['ctGAN']- all_c_index_dataframe['none']
        if cancer == 'LGG':
            first_model_all_c_index_df_LGG = all_c_index_dataframe['ctGAN']- all_c_index_dataframe['none']
        if cancer == 'LIHC':
            first_model_all_c_index_df_LIHC = all_c_index_dataframe['ctGAN']- all_c_index_dataframe['none']
        if cancer == 'LUAD':
            first_model_all_c_index_df_LUAD = all_c_index_dataframe['ctGAN']- all_c_index_dataframe['none']
        if cancer == 'LUSC':
            first_model_all_c_index_df_LUSC = all_c_index_dataframe['ctGAN']- all_c_index_dataframe['none']
        if cancer == 'OV':
            first_model_all_c_index_df_OV = all_c_index_dataframe['ctGAN']- all_c_index_dataframe['none']
        if cancer == 'SKCM':
            first_model_all_c_index_df_SKCM = all_c_index_dataframe['ctGAN']- all_c_index_dataframe['none']
        if cancer == 'STAD':
            first_model_all_c_index_df_STAD = all_c_index_dataframe['ctGAN']- all_c_index_dataframe['none']

    all_models_c_index_dataframe = pd.concat(
        [first_model_all_c_index_df_BLCA,
         first_model_all_c_index_df_COAD,
         first_model_all_c_index_df_HNSC,
         first_model_all_c_index_df_KIRC,
         first_model_all_c_index_df_LGG,
         first_model_all_c_index_df_LIHC,
         first_model_all_c_index_df_LUAD,
         first_model_all_c_index_df_LUSC,
         first_model_all_c_index_df_OV,
         first_model_all_c_index_df_SKCM,
         first_model_all_c_index_df_STAD,], axis=0)
    all_models_c_index_dataframe = np.asarray(all_models_c_index_dataframe).reshape(-1, 1)
    all_models_c_index_dataframe = pd.DataFrame(all_models_c_index_dataframe)
    final_all_models_c_index_df = pd.concat([all_models_c_index_dataframe, cancer_type_list], axis=1)
    final_all_models_c_index_df = pd.concat([final_all_models_c_index_df, model_type_list], axis=1)
    final_all_models_c_index_df.columns = ['log_rank_p_value', 'cancer', 'model']

    plt.figure(figsize=(10, 4))
    sns.set_style('whitegrid')
    sns.set_style('ticks')

    my_pal = {model_name_list[0]: "salmon"}

    ax = sns.boxplot(width=0.4, showfliers=False, data=final_all_models_c_index_df, palette=my_pal,
                     medianprops=dict(color='k', linewidth=1.2), capprops=dict(color='k', linewidth=0.8),
                     whiskerprops=dict(color='k', linewidth=0.8), fliersize=1.0, hue='model', x='cancer',
                     y='log_rank_p_value')

    for i in range(len(cancer_type) * number_models):
        mybox = ax.artists[i]
        mybox.set_edgecolor('black')
        mybox.set_linewidth(0.8)
        mybox.set_alpha(0.8)

    plt.ylabel("-log10(Log-rank P-value)", fontsize=12)
    plt.xlabel("", fontsize=3)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=8.5)
    plt.axhline(y=0.0, color='red', linewidth=1.0, linestyle='--')
    plt.legend(fontsize=10, loc='upper right')

    #plt.show()
    os.makedirs(f_name + '/gene_' + str(significant_gene_num) + '_' + 'epoch_' + str(train_epoch) + '/' , exist_ok=True)
    plt.savefig(f_name + '/gene_' + str(significant_gene_num) + '_' + 'epoch_' + str(train_epoch) + '/'  + '(11 cancers)ctGAN_improvement_kaplan meier_logrank_p_value_boxplot' + '.tiff', dpi=350)

def boxplot_one_cancer_improvement_log_rank_p_value(cancer = 'COAD', significant_gene_num = 300, train_epoch = 500): ### plot for 11 cancers, only use when augmentation = 'yes'

    f_name = './../kaplan_meier_evaluation/boxplot'

    os.makedirs(f_name, exist_ok=True)

    cancer_type_list = np.repeat(cancer, 30)
    cancer_type_list = cancer_type_list.reshape(-1, 1)
    cancer_type_list = pd.DataFrame(cancer_type_list)

    model_type_list = np.repeat('ctGAN', 30)
    model_type_list = model_type_list.reshape(-1, 1)
    model_type_list = pd.DataFrame(model_type_list)

    with open('./../kaplan_meier_evaluation/' + cancer + '/' + 'gene_' + str(significant_gene_num) + '_' + 'epoch_' + str(train_epoch) + '/no_augmentation/' + '30_logrank_pvalue_dataframe', "rb") as file:
        only_real_logrank_pvalue_list = pickle.load(file)

    with open('./../kaplan_meier_evaluation/' + cancer + '/' + 'gene_' + str(significant_gene_num) + '_' + 'epoch_' + str(train_epoch) + '/yes_augmentation/' + '30_logrank_pvalue_dataframe',"rb") as file:
        first_model_real_fake_logrank_pvalue_list = pickle.load(file)


    new_only_real_logrank_pvalue_list = []
    for i in range(30):
        value = (math.log10(only_real_logrank_pvalue_list[i]))*-1
        new_only_real_logrank_pvalue_list.append(value)

    new_first_model_real_fake_logrank_pvalue_list = []
    for i in range(30):
        value = (math.log10(first_model_real_fake_logrank_pvalue_list[i]))*-1
        new_first_model_real_fake_logrank_pvalue_list.append(value)


    new_only_real_logrank_pvalue_list_df = pd.DataFrame(new_only_real_logrank_pvalue_list)
    new_first_model_real_fake_logrank_pvalue_df = pd.DataFrame(new_first_model_real_fake_logrank_pvalue_list)

    all_c_index_dataframe = pd.concat([new_only_real_logrank_pvalue_list_df,  new_first_model_real_fake_logrank_pvalue_df], axis=1)
    all_c_index_dataframe.columns = ['none', 'ctGAN']



    first_model_all_c_index_df = all_c_index_dataframe['ctGAN'] - all_c_index_dataframe['none']


    all_models_c_index_dataframe = first_model_all_c_index_df
    all_models_c_index_dataframe = np.asarray(all_models_c_index_dataframe).reshape(-1, 1)
    all_models_c_index_dataframe = pd.DataFrame(all_models_c_index_dataframe)
    final_all_models_c_index_df = pd.concat([all_models_c_index_dataframe, cancer_type_list], axis=1)
    final_all_models_c_index_df = pd.concat([final_all_models_c_index_df, model_type_list], axis=1)
    final_all_models_c_index_df.columns = ['log_rank_p_value', 'cancer', 'model']

    plt.figure(figsize=(5, 5))
    sns.set_style('whitegrid')
    sns.set_style('ticks')

    my_pal = {'ctGAN': "salmon"}

    ax = sns.boxplot(width=0.2, showfliers=False, data=final_all_models_c_index_df, palette=my_pal,
                     medianprops=dict(color='k', linewidth=1.2), capprops=dict(color='k', linewidth=0.8),
                     whiskerprops=dict(color='k', linewidth=0.8), fliersize=1.0, hue='model', x='cancer',
                     y='log_rank_p_value')


    mybox = ax.artists[0]
    mybox.set_edgecolor('black')
    mybox.set_linewidth(0.8)
    mybox.set_alpha(0.8)

    plt.ylabel("-log10(Log-rank P-value)", fontsize=12)
    plt.xlabel("", fontsize=4)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=8.5)
    plt.axhline(y=0.0, color='red', linewidth=1.0, linestyle='--')
    plt.legend(fontsize=9, loc='upper right')

    #plt.show()
    os.makedirs(f_name + '/gene_' + str(significant_gene_num) + '_' + 'epoch_' + str(train_epoch) + '/', exist_ok=True)
    plt.savefig(f_name + '/gene_' + str(significant_gene_num) + '_' + 'epoch_' + str(train_epoch) + '/'  + '(' + cancer + ')' + 'ctGAN_improvement_kaplan meier_logrank_p_value_boxplot' + '.tiff', dpi=350)

