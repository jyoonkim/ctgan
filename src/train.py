import numpy as np
import itertools
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import random

from models import Generator, Discriminator
from augmentdata import GeneDataset
from util import ReplayBuffer, Logger, initialize_weights
from train_data_preprocess import train_dataset_load

def model_train(cancer_type = 'COAD', significant_gene_num = 300, epoch_num = 500, g_hidden_node_1 = 512, g_hidden_node_2 = 512, g_residual_block = 3, d_hidden_node_1 = 64, d_hidden_node_2 = 64):
    #cancer_type = 'COAD'  ### It could be other types of cancer ('BLCA', 'HNSC', 'KIRC', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'SKCM', 'STAD')

    batch_num = 16
    learning_rate_G = 0.0002
    learning_rate_D = 0.0001

    discriminator_ratio = 1
    generator_ratio = 3

    random_seed = 1004
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Final_TCGA, Final_TCGA2 = train_dataset_load(cancer_type=cancer_type, significant_gene_num=significant_gene_num)

    A_train_data, A_test_data = train_test_split(Final_TCGA, test_size=0.1, shuffle=True, random_state=0)
    B_train_data, B_test_data = train_test_split(Final_TCGA2, test_size=0.1, shuffle=True, random_state=0)

    original_A_train = A_train_data.copy()
    A_train_data = (A_train_data - A_train_data.mean()) / (A_train_data.std() + 1.2e-17)

    original_B_train = B_train_data.copy()
    B_train_data = (B_train_data - B_train_data.mean()) / (B_train_data.std() + 1.2e-17)

    A_train_data = A_train_data.to_numpy()
    A_train_data = torch.FloatTensor(A_train_data)

    B_train_data = B_train_data.to_numpy()
    B_train_data = torch.FloatTensor(B_train_data)

    input_dim = significant_gene_num + 1
    output_dim = significant_gene_num + 1

    train_dataloader = DataLoader(GeneDataset(A_train_data, B_train_data, unaligned=True), batch_size=batch_num, shuffle=True, drop_last=True)

    netG_A2B = Generator(input_dim, output_dim, hidden_node_1=g_hidden_node_1, hidden_node_2=g_hidden_node_2, n_residual_blocks=g_residual_block)
    netG_B2A = Generator(output_dim, input_dim, hidden_node_1=g_hidden_node_1, hidden_node_2=g_hidden_node_2, n_residual_blocks=g_residual_block)
    netD_A = Discriminator(input_dim, hidden_node_1=d_hidden_node_1, hidden_node_2=d_hidden_node_2)
    netD_B = Discriminator(output_dim, hidden_node_1=d_hidden_node_1, hidden_node_2=d_hidden_node_2)

    netG_A2B = netG_A2B.to(device)
    netG_B2A = netG_B2A.to(device)
    netD_A = netD_A.to(device)
    netD_B = netD_B.to(device)

    netG_A2B.train()
    netG_B2A.train()
    netD_A.train()
    netD_B.train()

    netG_A2B.apply(initialize_weights)
    netG_B2A.apply(initialize_weights)
    netD_A.apply(initialize_weights)
    netD_B.apply(initialize_weights)

    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    criterion_variance = torch.nn.L1Loss()

    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr = learning_rate_G, betas=(0, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr = learning_rate_D, betas = (0, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=learning_rate_D, betas=(0, 0.999))

    target_real = torch.Tensor(batch_num,1).fill_(1.0).to(device)
    target_fake = torch.Tensor(batch_num,1).fill_(0.0).to(device)

    ### ReplayBuffer
    fake_A_buffer = ReplayBuffer()  ### To address the instability of GAN, the method involves periodically presenting data previously generated by the generator to the discriminator
    fake_B_buffer = ReplayBuffer()

    ## Logger
    logger = Logger(epoch_num, len(train_dataloader))

    for epoch in range(epoch_num):
        for batch_idx, inputdata in enumerate(train_dataloader):
            print("batch idx")
            print(batch_idx)

            real_A = inputdata['A'][:,0:input_dim]
            real_B = inputdata['B'][:,0:input_dim]

            real_A = real_A.to(device)
            real_B = real_B.to(device)

            if batch_idx == 0 or batch_idx % (discriminator_ratio + generator_ratio) >= discriminator_ratio:
                for _ in range(generator_ratio):

                    #### Generators A2B and B2A ########
                    optimizer_G.zero_grad()

                    # Identity loss
                    # G_A2B(B) should equal B if real B is fed
                    same_B = netG_A2B(real_B)
                    loss_identity_B = criterion_identity(same_B, real_B)*1.0

                    # G_B2A(A) should equal A if real A is fed
                    same_A = netG_B2A(real_A)
                    loss_identity_A = criterion_identity(same_A, real_A)*1.0

                    # GAN loss
                    fake_B = netG_A2B(real_A)
                    pred_fake = netD_B(fake_B)
                    loss_GAN_A2B = criterion_GAN(pred_fake, target_real)*1.0

                    fake_A = netG_B2A(real_B)
                    pred_fake = netD_A(fake_A)
                    loss_GAN_B2A = criterion_GAN(pred_fake, target_real)*1.0

                    # Cycle loss
                    recovered_A = netG_B2A(fake_B)
                    loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*1.0

                    recovered_B = netG_A2B(fake_A)
                    loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*1.0

                    loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

                    loss_G.backward()
                    optimizer_G.step()

                # Progress report (http://localhost:8097)
                if batch_idx>=0 and batch_idx<=3:
                    logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B),'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                                'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB)})
                else:
                    logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B),'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)})

            else:
                ######## Discriminator A ############
                optimizer_D_A.zero_grad()

                # Real loss
                pred_real = netD_A(real_A)
                loss_D_real = criterion_GAN(pred_real, target_real)

                # Fake loss
                fake_A = fake_A_buffer.push_and_pop(fake_A)
                pred_fake = netD_A(fake_A.detach())
                loss_D_fake = criterion_GAN(pred_fake, target_fake)

                # Total loss
                loss_D_A = (loss_D_real + loss_D_fake)*0.5
                loss_D_A.backward()

                optimizer_D_A.step()

                ###### Discriminator B ############
                optimizer_D_B.zero_grad()

                # Real loss
                pred_real = netD_B(real_B)
                loss_D_real = criterion_GAN(pred_real, target_real)

                # Fake loss
                fake_B = fake_B_buffer.push_and_pop(fake_B)
                pred_fake = netD_B(fake_B.detach())
                loss_D_fake = criterion_GAN(pred_fake, target_fake)

                # Total loss
                loss_D_B = (loss_D_real+ loss_D_fake)*0.5
                loss_D_B.backward()
                optimizer_D_B.step()

                ###################################

                # Progress report (http://localhost:8097)
                logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B),'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)})

        os.makedirs('./../saved_models/'+cancer_type+'/gene_'+str(significant_gene_num),exist_ok=True)

        torch.save(netG_A2B.state_dict(), './../saved_models/'+cancer_type+'/gene_'+str(significant_gene_num)+'/netG_A2B.pt')
        torch.save(netG_B2A.state_dict(), './../saved_models/'+cancer_type+'/gene_'+str(significant_gene_num)+'/netG_B2A.pt')
        torch.save(netD_A.state_dict(), './../saved_models/'+cancer_type+'/gene_'+str(significant_gene_num)+'/netD_A.pt')
        torch.save(netD_B.state_dict(), './../saved_models/'+cancer_type+'/gene_'+str(significant_gene_num)+'/netD_B.pt')

        if ((epoch+1)%100)==0:
            torch.save(netG_A2B.state_dict(), './../saved_models/'+cancer_type+'/gene_'+str(significant_gene_num)+'/netG_A2B_'+str(epoch)+'.pt')
            torch.save(netG_B2A.state_dict(), './../saved_models/'+cancer_type+'/gene_'+str(significant_gene_num)+'/netG_B2A_'+str(epoch)+'.pt')
            torch.save(netD_A.state_dict(), './../saved_models/'+cancer_type+'/gene_'+str(significant_gene_num)+'/netD_A_'+str(epoch)+'.pt')
            torch.save(netD_B.state_dict(), './../saved_models/'+cancer_type+'/gene_'+str(significant_gene_num)+'/netD_B_'+str(epoch)+'.pt')


