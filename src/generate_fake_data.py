import pickle
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from models import Generator
from train_data_preprocess import train_dataset_load


def make_fake_data(cancer_type = ['COAD'], significant_gene_num = 300, train_epoch = 500):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #cancer_type = ['COAD']  ## If you want all cancer types -> ['BLCA','COAD', 'HNSC', 'KIRC', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'SKCM', 'STAD']

    DIR = './../saved_models/'

    for cancer in cancer_type:
        Final_TCGA, Final_TCGA2 = train_dataset_load(cancer_type=cancer, significant_gene_num=significant_gene_num)

        A_train_data, A_test_data = train_test_split(Final_TCGA, test_size=0.1, shuffle=True, random_state=0)
        B_train_data, B_test_data = train_test_split(Final_TCGA2, test_size=0.1, shuffle=True, random_state=0)

        original_A_train = A_train_data.copy()
        original_B_train = B_train_data.copy()

        Final_TCGA = (Final_TCGA - original_A_train.mean()) / (original_A_train.std() + 1.2e-17)
        Final_TCGA2 = (Final_TCGA2 - original_B_train.mean()) / (original_B_train.std() + 1.2e-17)

        Final_TCGA = Final_TCGA.to_numpy()
        Final_TCGA = torch.FloatTensor(Final_TCGA)

        Final_TCGA2 = Final_TCGA2.to_numpy()
        Final_TCGA2 = torch.FloatTensor(Final_TCGA2)

        input_dim = significant_gene_num + 1
        output_dim = significant_gene_num + 1

        All_A_data_loader = DataLoader(Final_TCGA, batch_size=Final_TCGA.shape[0], shuffle=False, drop_last=False) #### BLCA
        All_B_data_loader = DataLoader(Final_TCGA2, batch_size=Final_TCGA2.shape[0], shuffle=False, drop_last=False) ##### BRCA

        netG_A2B= Generator(input_dim, output_dim)
        netG_B2A = Generator(output_dim, input_dim)

        netG_A2B= netG_A2B.to(device)
        netG_B2A = netG_B2A.to(device)

        netG_A2B.load_state_dict(torch.load(DIR+cancer+'/gene_'+str(significant_gene_num)+'/netG_A2B_'+str(train_epoch-1)+'.pt'))
        netG_B2A.load_state_dict(torch.load(DIR+cancer+'/gene_'+str(significant_gene_num)+'/netG_B2A_'+str(train_epoch-1)+'.pt'))

        netG_A2B.eval()
        netG_B2A.eval()

        for batch_idx, inputdata in enumerate(All_A_data_loader):
            # print("batch idx")
            # print(batch_idx)
            real_A = inputdata[:, 0:input_dim]

            real_A = real_A.to(device)

            fake_B = netG_A2B(real_A)

        with open(DIR+cancer+'/gene_'+str(significant_gene_num)+'/All_fake_B_'+str(train_epoch), "wb") as file:
            pickle.dump(fake_B, file)

        #####################################

        netG_A2B= Generator(input_dim, output_dim)
        netG_B2A = Generator(output_dim, input_dim)

        netG_A2B= netG_A2B.to(device)
        netG_B2A = netG_B2A.to(device)

        netG_A2B.load_state_dict(torch.load(DIR+cancer+'/gene_'+str(significant_gene_num)+'/netG_A2B_'+str(train_epoch-1)+'.pt'))
        netG_B2A.load_state_dict(torch.load(DIR+cancer+'/gene_'+str(significant_gene_num)+'/netG_B2A_'+str(train_epoch-1)+'.pt'))

        netG_A2B.eval()
        netG_B2A.eval()

        for batch_idx, inputdata in enumerate(All_B_data_loader):
            # print("batch idx")
            # print(batch_idx)
            real_B = inputdata[:, 0:input_dim]

            real_B = real_B.to(device)

            fake_A = netG_B2A(real_B)


        with open(DIR+cancer+'/gene_'+str(significant_gene_num)+'/All_fake_A_'+str(train_epoch), "wb") as file:
            pickle.dump(fake_A, file)
