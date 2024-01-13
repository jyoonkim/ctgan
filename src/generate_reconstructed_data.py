import pickle
import torch
from torch.utils.data import DataLoader
from models import Generator


def make_reconstructed_data(cancer_type = ['COAD'],significant_gene_num = 300, train_epoch = 500):

    # cancer_type = ['COAD'] ## If you want all cancer types -> ['BLCA','COAD', 'HNSC', 'KIRC', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'SKCM', 'STAD']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    DIR = './../saved_models/'

    for cancer in cancer_type:

        input_dim = significant_gene_num+1
        output_dim = significant_gene_num+1

        with open(DIR+cancer+'/gene_'+str(significant_gene_num)+'/All_fake_B_'+str(train_epoch), "rb") as file:
            fake_B = pickle.load(file)

        with open(DIR+cancer+'/gene_'+str(significant_gene_num)+'/All_fake_A_'+str(train_epoch), "rb") as file:
            fake_A = pickle.load(file)

        fake_B = fake_B.cpu().detach().numpy()
        fake_A = fake_A.cpu().detach().numpy()

        fake_B = torch.FloatTensor(fake_B)
        fake_A = torch.FloatTensor(fake_A)

        All_fake_A_data_loader = DataLoader(fake_A, batch_size=fake_A.shape[0], shuffle=False, drop_last=False)
        All_fake_B_data_loader = DataLoader(fake_B, batch_size=fake_B.shape[0], shuffle=False, drop_last=False)

        netG_A2B= Generator(input_dim, output_dim)
        netG_B2A = Generator(output_dim, input_dim)

        netG_A2B= netG_A2B.to(device)
        netG_B2A = netG_B2A.to(device)

        netG_A2B.load_state_dict(torch.load(DIR+cancer+'/gene_'+str(significant_gene_num)+'/netG_A2B_'+str(train_epoch-1)+'.pt'))
        netG_B2A.load_state_dict(torch.load(DIR+cancer+'/gene_'+str(significant_gene_num)+'/netG_B2A_'+str(train_epoch-1)+'.pt'))

        netG_A2B.eval()
        netG_B2A.eval()

        for batch_idx, inputdata in enumerate(All_fake_A_data_loader):
            # print("batch idx")
            # print(batch_idx)
            fake_A = inputdata[:, 0:input_dim]

            fake_A = fake_A.to(device)

            reconstruct_fake_B = netG_A2B(fake_A)

        with open(DIR+cancer+'/gene_'+str(significant_gene_num)+'/All_reconstruction_B_'+str(train_epoch), "wb") as file:
            pickle.dump(reconstruct_fake_B, file)

        #####################################

        netG_A2B= Generator(input_dim, output_dim)
        netG_B2A = Generator(output_dim, input_dim)

        netG_A2B= netG_A2B.to(device)
        netG_B2A = netG_B2A.to(device)

        netG_A2B.load_state_dict(torch.load(DIR+cancer+'/gene_'+str(significant_gene_num)+'/netG_A2B_'+str(train_epoch-1)+'.pt'))
        netG_B2A.load_state_dict(torch.load(DIR+cancer+'/gene_'+str(significant_gene_num)+'/netG_B2A_'+str(train_epoch-1)+'.pt'))

        netG_A2B.eval()
        netG_B2A.eval()

        for batch_idx, inputdata in enumerate(All_fake_B_data_loader):
            # print("batch idx")
            # print(batch_idx)
            fake_B = inputdata[:, 0:input_dim]

            fake_B = fake_B.to(device)

            reconstruct_fake_A = netG_B2A(fake_B)


        with open(DIR+cancer+'/gene_'+str(significant_gene_num)+'/All_reconstruction_A_'+str(train_epoch), "wb") as file:
            pickle.dump(reconstruct_fake_A, file)
