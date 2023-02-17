
# In[1]:


import torch
import torch.nn as nn
from torchvision.transforms import transforms
# import numpy as np
from tqdm import tqdm
# import random
from torch.utils.data import Dataset, DataLoader


# In[2]:


DEVICE = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
EPOCHS = 2
TIMESTEP = 2000
LR = 3e-4
BATCH_SIZE = 5
NUM_WORKERS = 2
BETA1 = 0.5
BETA2 = 0.999
SAVE_MODEL = True
LOAD_MODEL = False
NET1_CHK = "./net1.pth.tar"
NET2_CHK = "./net2.pth.tar"
NETCAT_CHK = "./netcat.pth.tar"


# In[3]:


class WaveDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.input = []
        self.output = []
        
        for _ in range(20):
            self.input.append(torch.randn(2, 1925))
            self.output.append(torch.rand(1))
    
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, index):
        return {'feature': self.input[index], 'target': self.output[index]}


# In[4]:


class Convolute(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size, dropout=0.0, maxpool=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_filters, out_filters, kernel_size),
            nn.BatchNorm1d(out_filters),
            nn.ReLU()
        )
        if dropout!=0: self.conv.append(nn.Dropout(dropout))
        if maxpool!=0: self.conv.append(nn.MaxPool1d(maxpool))

    def forward(self, x):
        return self.conv(x)

class DNF(nn.Module):
    def __init__(self, in_filters, out_filters, do_transpose=False, do_flatten=False):
        super().__init__()
        self.do_transpose = do_transpose
        self.dnf = nn.Sequential(nn.Linear(in_filters, out_filters))
        if do_flatten: self.dnf.append(nn.Flatten())
    
    def forward(self, x):
        if self.do_transpose: x = torch.transpose(x, 2, 1)
        return self.dnf(x)


class ANN(nn.Module):
    def __init__(self, in_filters, hidden_filters=50):
        super().__init__()
        self.ann = nn.Sequential(
            nn.Linear(in_filters, hidden_filters),
            nn.ReLU(),
            nn.Linear(hidden_filters, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.ann(x)
    


# In[5]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Convolute(2, 576, 11)
        self.conv2 = Convolute(576, 484, 11, 0.3, 4)
        self.conv3 = Convolute(484, 400, 5)
        self.conv4 = Convolute(400, 324, 5, 0.2)
        self.conv5 = DNF(324, 256, True, True)
        self.conv6 = DNF(119808, 150)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return self.conv6(x)

class NetCat(nn.Module):
    def __init__(self):
        super().__init__()
        self.ann1 = ANN(300)

    def forward(self, x):
        return self.ann1(x)


# In[6]:


def save_checkpoint(model, optim, filename="/checkpoint.path.tar"):
    print("=> Saving Checkpoint")
    checkpoint = {
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optim, lr):
    print("=> Loading Checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    optim.load_state_dict(checkpoint["optim_state"])
    for group in optim.param_groups:
        group["lr"] = lr


# In[7]:


def train(data_loader, net1, net2, netcat, net1_scaler, net2_scaler, netcat_scaler, net1_optim, net2_optim, netcat_optim, loss_function):
    losses1 = []
    losses2 = []
    lossescat = []
    data_loop = tqdm(data_loader, leave=True)

    net1.zero_grad()
    net2.zero_grad()
    netcat.zero_grad()

    for data in data_loop:
        features, targets = data['feature'].to(DEVICE), data['target'].to(DEVICE)

        with torch.cuda.amp.autocast():
            dense1 = net1(features)
            dense2 = net2(features)
            dense = torch.cat([dense1, dense2], dim = 1)
            preds = netcat(dense)
            loss1 = loss_function(preds, targets)
            loss2 = loss_function(preds, targets)
            losscat = loss_function(preds, targets)

        netcat.zero_grad()
        netcat_scaler.scale(losscat).backward(retain_graph=True)
        netcat_scaler.step(netcat_optim)
        netcat_scaler.update()

        net2.zero_grad()
        net2_scaler.scale(loss2).backward(retain_graph=True)
        net2_scaler.step(net2_optim)
        net2_scaler.update()
        
        net1.zero_grad()
        net1_scaler.scale(loss1).backward(retain_graph=True)
        net1_scaler.step(net1_optim)
        net1_scaler.update()

        losses1.append(loss1.data)
        losses2.append(loss2.data)
        lossescat.append(losscat.data)

    loss_avg = torch.mean(torch.FloatTensor(losses1))
    print(f'Average Loss this epoch = {loss_avg}')


def main():
    net1, net2, netcat = Net(), Net(), NetCat()
    net1, net2, netcat = net1.to(DEVICE), net2.to(DEVICE), netcat.to(DEVICE)
    net1_scaler, net2_scaler, netcat_scaler = torch.cuda.amp.GradScaler(), torch.cuda.amp.GradScaler(), torch.cuda.amp.GradScaler()
    net1_optim = torch.optim.Adam(net1.parameters(), lr=LR, betas=(BETA1, BETA2))
    net2_optim = torch.optim.Adam(net2.parameters(), lr=LR, betas=(BETA1, BETA2))
    netcat_optim = torch.optim.Adam(netcat.parameters(), lr=LR, betas=(BETA1, BETA2))
    loss_function = nn.BCEWithLogitsLoss()
    
    
    train_data = WaveDataset()
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    if LOAD_MODEL:
        load_checkpoint(NET1_CHK, net1, net1_optim, lr=LR)
        load_checkpoint(NET2_CHK, net2, net2_optim, lr=LR)
        load_checkpoint(NETCAT_CHK, netcat, netcat_optim, lr=LR)

    for epoch in range(EPOCHS):
        print(f'Epoch count = {epoch+1}')
        train(train_loader, net1, net2, netcat, net1_scaler, net2_scaler, netcat_scaler, net1_optim, net2_optim, netcat_optim, loss_function)
        if SAVE_MODEL and (epoch+1) % 5 == 0:
            save_checkpoint(net1, net1_optim, filename=NET1_CHK)
            save_checkpoint(net2, net2_optim, filename=NET2_CHK)
            save_checkpoint(netcat, netcat_optim, filename=NETCAT_CHK)

if __name__ == "__main__":
    main()


# x1=torch.randn(1, 2, 1925)
# head1 = []
# head1.append(Convolute(2, 576, 11))
# head1.append(Convolute(576, 484, 11, 0.3, 4))
# head1.append(Convolute(484, 400, 5))
# head1.append(Convolute(400, 324, 5, 0.2))
# head1.append(DNF(324, 256, True, True))
# head1.append(DNF(119808, 150))
# for obj in head1: x1=obj(x1)
# x1.shape

# x2=torch.randn(1, 2, 1925)
# head2 = []
# head2.append(Convolute(2, 576, 11))
# head2.append(Convolute(576, 484, 11, 0.3, 4))
# head2.append(Convolute(484, 400, 5))
# head2.append(Convolute(400, 324, 5, 0.2))
# head2.append(DNF(324, 256, True, True))
# head2.append(DNF(119808, 150))
# for obj in head2: x2=obj(x2)
# x2.shape

# x = torch.cat([x1, x2], dim=1)
# x.shape

