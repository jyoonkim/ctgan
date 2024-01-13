import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        linear_block = [ nn.Linear(in_features, in_features),
                         nn.BatchNorm1d(in_features),
                         nn.ReLU(inplace=True),
                         nn.Linear(in_features, in_features),
                         nn.BatchNorm1d(in_features)]

        self.linear_block = nn.Sequential(*linear_block)

    def forward(self,x):
        return x + self.linear_block(x)

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_node_1=512, hidden_node_2=512, n_residual_blocks = 3):
        super(Generator, self).__init__()

        # Initial Linear block
        model = [ nn.Linear(input_dim, hidden_node_1),
                  nn.BatchNorm1d(hidden_node_1),
                  nn.ReLU(inplace=True) ]

        # Linear block
        model += [nn.Linear(hidden_node_1,hidden_node_2),
                  nn.BatchNorm1d(hidden_node_2),
                  nn.ReLU(inplace=True)]

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(hidden_node_2)]

        model += [nn.Linear(hidden_node_2,hidden_node_1),
                  nn.BatchNorm1d(hidden_node_1),
                  nn.ReLU(inplace=True)]

        model += [nn.Linear(hidden_node_1, hidden_node_1),
                  nn.BatchNorm1d(hidden_node_1),
                  nn.ReLU(inplace=True),
                  nn.Linear(hidden_node_1, output_dim)
                  ]

        self.model = nn.Sequential(*model)


    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self,  input_dim, hidden_node_1=64, hidden_node_2=64):
        super(Discriminator, self).__init__()

        model = [ nn.Linear(input_dim,hidden_node_1),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [ nn.Linear(hidden_node_1, hidden_node_2),
                   nn.LeakyReLU(0.2, inplace=True)]

        model += [ nn.Linear(hidden_node_2, hidden_node_2),
                   nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Linear(hidden_node_2,1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x





