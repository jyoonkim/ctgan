import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
matplotlib.font_manager._rebuild()
from sklearn.manifold import TSNE
import os

from evaluate_data_preprocess import evaluate_dataset_load

def tsne_plot(cancer_type = ['COAD'], significant_gene_num  = 300, train_epoch = 500):
    #cancer_type = ['COAD']  ## If you want all cancer types -> ['BLCA', 'COAD','HNSC', 'KIRC', 'LGG','LIHC', 'LUAD', 'LUSC', 'OV', 'SKCM', 'STAD']

    for cancer in cancer_type:

        Final_TCGA, fake_A, reconstruct_A, Final_TCGA2, fake_B, reconstruct_B = evaluate_dataset_load(cancer = cancer, DIR ='./../saved_models/', significant_gene_num  = significant_gene_num, train_epoch = train_epoch)

        f_name = './../tsne_plot_evaluation/' + cancer + '/' + 'gene_' + str(significant_gene_num) + '_' + 'epoch_' + str(train_epoch) +'/'

        os.makedirs(f_name, exist_ok=True)

    #### 1. real and generated data for [cancer] and BRCA
        Total_TCGA = pd.concat([fake_A, Final_TCGA, fake_B,  Final_TCGA2], axis=0)
        Total_TCGA = Total_TCGA.iloc[:,0:significant_gene_num]

        string_1 = 'generated ' + cancer
        string_2 = 'real ' + cancer
        string_3 = 'generated BRCA'
        string_4 = 'real BRCA'

        tsne = TSNE(random_state=42, perplexity=50)
        Total_tcga_gene = Total_TCGA

        Total_tcga_label = []
        Total_tcga_real_label = []
        for i in range(fake_A.shape[0]):
            Total_tcga_label.append('orange')
            Total_tcga_real_label.append(string_1)
        for i in range(Final_TCGA.shape[0]):
            Total_tcga_label.append('steelblue')
            Total_tcga_real_label.append(string_2)
        for i in range(fake_B.shape[0]):
            Total_tcga_label.append('hotpink')
            Total_tcga_real_label.append(string_3)
        for i in range(Final_TCGA2.shape[0]):
            Total_tcga_label.append('limegreen')
            Total_tcga_real_label.append(string_4)

        Total_tsne = tsne.fit_transform(Total_tcga_gene)

        tsne_results = pd.DataFrame(Total_tsne, columns=['tsne1', 'tsne2']).assign(category=Total_tcga_real_label).assign(color=Total_tcga_label).groupby('category')

        # for name, points in tsne_results:
        #    print(name)
        #    print(points)

        plt.figure(figsize=(10, 10))
        plt.xlim(Total_tsne[:, 0].min()-10, Total_tsne[:, 0].max() + 10)
        plt.ylim(Total_tsne[:, 1].min()-10, Total_tsne[:, 1].max() + 10)

        plt.rc('font', family='Malgun Gothic')
        plt.rc('axes', unicode_minus=False)
        for name, points in tsne_results:
           plt.scatter(points.tsne1, points.tsne2, c=points.color, label=name, s= 4.5**2)
        plt.legend(fontsize=17, loc='lower right')
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        #plt.show()
        plt.savefig(f_name +'(' + cancer+')' +'real_fake_with BRCA.tiff', dpi=350)

    #### 2. real and reconstructed data for [cancer]
        Total_TCGA = pd.concat([Final_TCGA, reconstruct_A], axis=0)
        Total_TCGA = Total_TCGA.iloc[:, 0:significant_gene_num]

        string_1 = 'real ' + cancer
        string_2 = 'reconstruct '+ cancer
        string_3 = ''
        string_4 = ''

        tsne = TSNE(random_state=42, perplexity=50)
        Total_tcga_gene = Total_TCGA
        Total_tcga_label = []
        Total_tcga_real_label = []

        if cancer=='BLCA': color_string = 'blue'
        if cancer=='COAD': color_string = 'black'
        if cancer=='HNSC': color_string = 'green'
        if cancer=='KIRC': color_string = 'c'
        if cancer=='LGG': color_string = 'salmon'
        if cancer=='LIHC': color_string = 'red'
        if cancer=='LUAD': color_string = 'darkorchid'
        if cancer=='LUSC': color_string = 'lightskyblue'
        if cancer=='OV': color_string = 'gold'
        if cancer=='SKCM': color_string = 'orange'
        if cancer=='STAD': color_string = 'saddlebrown'

        for i in range(Final_TCGA.shape[0]):
            Total_tcga_label.append('steelblue')
            Total_tcga_real_label.append(string_1)
        for i in range(reconstruct_A.shape[0]):
            Total_tcga_label.append('orange')
            Total_tcga_real_label.append(string_2)

        Total_tsne = tsne.fit_transform(Total_tcga_gene)

        tsne_results = pd.DataFrame(Total_tsne, columns=['tsne1', 'tsne2']).assign(
            category=Total_tcga_real_label).assign(color=Total_tcga_label).groupby('category')

        # for name, points in tsne_results:
        #     print(name)
        #     print(points)

        plt.figure(figsize=(10, 10))
        plt.xlim(Total_tsne[:, 0].min() -10, Total_tsne[:, 0].max() + 10)
        plt.ylim(Total_tsne[:, 1].min() -10, Total_tsne[:, 1].max() + 10)

        plt.rc('font', family='Malgun Gothic')
        plt.rc('axes', unicode_minus=False)

        for name, points in tsne_results:
            if 'reconstruct' not in name:
                plt.scatter(points.tsne1, points.tsne2, c=points.color,s=12 ** 2, marker='.',  label='real ' + cancer)
            else:
                plt.scatter(points.tsne1, points.tsne2, c=points.color, s=10 ** 2, marker='+', label='reconstruct ' + cancer)

        plt.legend(fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        plt.savefig(f_name +'(' + cancer+')' +'real_reconstruct.tiff', dpi=350)

    ####3. real and reconstructed data for [cancer] and BRCA

        Total_TCGA = pd.concat([Final_TCGA, reconstruct_A, Final_TCGA2, reconstruct_B], axis=0)
        Total_TCGA = Total_TCGA.iloc[:, 0:significant_gene_num]

        string_1 = 'real ' + cancer
        string_2 = 'reconstruct ' + cancer
        string_3 = 'real BRCA'
        string_4 = 'reconstruct BRCA'

        tsne = TSNE(random_state=42, perplexity=50)
        Total_tcga_gene = Total_TCGA
        Total_tcga_label = []
        Total_tcga_real_label = []

        for i in range(Final_TCGA.shape[0]):
            Total_tcga_label.append('steelblue')
            Total_tcga_real_label.append(string_1)
        for i in range(reconstruct_A.shape[0]):
            Total_tcga_label.append('orange')
            Total_tcga_real_label.append(string_2)
        for i in range(Final_TCGA2.shape[0]):
            Total_tcga_label.append('limegreen')
            Total_tcga_real_label.append(string_3)
        for i in range(reconstruct_B.shape[0]):
            Total_tcga_label.append('hotpink')
            Total_tcga_real_label.append(string_4)

        Total_tsne = tsne.fit_transform(Total_tcga_gene)

        tsne_results = pd.DataFrame(Total_tsne, columns=['tsne1', 'tsne2']).assign(
            category=Total_tcga_real_label).assign(color=Total_tcga_label).groupby('category')

        # for name, points in tsne_results:
        #     print(name)
        #     print(points)

        plt.figure(figsize=(10, 10))
        plt.xlim(Total_tsne[:, 0].min()-10, Total_tsne[:, 0].max() + 10)
        plt.ylim(Total_tsne[:, 1].min()-10, Total_tsne[:, 1].max() + 10)

        plt.rc('font', family='Malgun Gothic')
        plt.rc('axes', unicode_minus=False)

        for name, points in tsne_results:
            if 'reconstruct' not in name:
                plt.scatter(points.tsne1, points.tsne2, c=points.color, s=10 ** 2, marker='.', label=name)
            else:
                plt.scatter(points.tsne1, points.tsne2, c=points.color, s=8 ** 2, marker='+', label=name)
        plt.legend(fontsize=17)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.savefig(f_name +'(' + cancer+')' +'real_reconstruct_with BRCA.tiff', dpi=350)

    ####### 4. real and fake data for [cancer]

        Total_TCGA = pd.concat([Final_TCGA, fake_A], axis=0)
        Total_TCGA = Total_TCGA.iloc[:,0:significant_gene_num]

        tsne = TSNE(random_state=42, perplexity=50)
        Total_tcga_gene = Total_TCGA
        Total_tcga_label = []
        Total_tcga_real_label = []

        if cancer=='BLCA': color_string = 'blue'
        if cancer=='COAD': color_string = 'black'
        if cancer=='HNSC': color_string = 'green'
        if cancer=='KIRC': color_string = 'c'
        if cancer=='LGG': color_string = 'salmon'
        if cancer=='LIHC': color_string = 'red'
        if cancer=='LUAD': color_string = 'darkorchid'
        if cancer=='LUSC': color_string = 'lightskyblue'
        if cancer=='OV': color_string = 'gold'
        if cancer=='SKCM': color_string = 'orange'
        if cancer=='STAD': color_string = 'saddlebrown'

        for i in range(Final_TCGA.shape[0]):
            Total_tcga_label.append('steelblue')
            Total_tcga_real_label.append('real ' + cancer)

        for i in range(fake_A.shape[0]):
            Total_tcga_label.append('orange')
            Total_tcga_real_label.append('generated ' + cancer)

        Total_tsne = tsne.fit_transform(Total_tcga_gene)

        tsne_results = pd.DataFrame(Total_tsne, columns=['tsne1', 'tsne2']).assign(category=Total_tcga_real_label).assign(color=Total_tcga_label).groupby('category')

        # for name, points in tsne_results:
        #    print(name)
        #    print(points)

        plt.figure(figsize=(10, 10))
        plt.xlim(Total_tsne[:, 0].min() - 10, Total_tsne[:, 0].max() + 10)
        plt.ylim(Total_tsne[:, 1].min() - 10, Total_tsne[:, 1].max() + 10)

        plt.rc('font', family='Malgun Gothic')
        plt.rc('axes', unicode_minus=False)
        for name, points in tsne_results:
            if 'generated' not in name:
                plt.scatter(points.tsne1, points.tsne2, c=points.color,s=11 ** 2, marker='.', label='real ' + cancer)
            else:
                plt.scatter(points.tsne1, points.tsne2, c=points.color, s=11 ** 2, marker='.', label='generated ' + cancer)
        plt.legend(fontsize=17.5)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        #plt.show()

        plt.savefig(f_name +'(' + cancer+')' +'real_fake.tiff', dpi=350)

