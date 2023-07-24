# Generator synthetic Running and Jumping data 
# Made them to a Pytorch Dataset 

from torch.utils.data import Dataset, DataLoader
import torch
from GANModels import *
import numpy as np
import os


# 合成数据，由GAN模型生成（对比用）
class Synthetic_Dataset(Dataset):
    def __init__(self,
                 # Jumping_model_path = './pre-trained-models/JumpingGAN_checkpoint',
                 # Running_model_path = './pre-trained-models/RunningGAN_checkpoint',
                 Jumping_model_path='./pre-trained-models/Jumping_0712',
                 Running_model_path='./pre-trained-models/Running_0712',
                 Shenzhen_model_path='./pre-trained-models/Shenzhen_10000',
                 data_mode='JumpingRunning',
                 sample_size=1000
                 ):
        assert data_mode == 'JumpingRunning' or data_mode == 'Shenzhen'

        self.sample_size = sample_size
        self.data_mode = data_mode

        # Generate Running Data
        running_gen_net = Generator(seq_len=150, channels=3, latent_dim=100)
        running_ckp = torch.load(Running_model_path, map_location=torch.device('cpu'))
        running_gen_net.load_state_dict(running_ckp['gen_state_dict'])

        # Generate Jumping Data
        jumping_gen_net = Generator(seq_len=150, channels=3, latent_dim=100)
        jumping_ckp = torch.load(Jumping_model_path, map_location=torch.device('cpu'))
        jumping_gen_net.load_state_dict(jumping_ckp['gen_state_dict'])

        # Generate Shenzhen Data
        shenzhen_gen_net = Generator(seq_len=64, channels=11, latent_dim=100)
        shenzhen_ckp = torch.load(Shenzhen_model_path, map_location=torch.device('cpu'))
        shenzhen_gen_net.load_state_dict(shenzhen_ckp['gen_state_dict'])

        # generate synthetic running data label is 0
        z = torch.FloatTensor(np.random.normal(0, 1, (self.sample_size, 100)))
        self.syn_running = running_gen_net(z)
        self.syn_running = self.syn_running.detach().numpy()
        self.running_label = np.zeros(len(self.syn_running))
        print(self.syn_running.shape)

        # generate synthetic jumping data label is 1
        z = torch.FloatTensor(np.random.normal(0, 1, (self.sample_size, 100)))
        self.syn_jumping = jumping_gen_net(z)
        self.syn_jumping = self.syn_jumping.detach().numpy()
        self.jumping_label = np.zeros(len(self.syn_jumping)) + 1
        print(self.syn_jumping.shape)

        # generate synthetic shenzhen data label is 7
        z = torch.FloatTensor(np.random.normal(0, 1, (self.sample_size, 100)))
        self.syn_shenzhen = shenzhen_gen_net(z)
        self.syn_shenzhen = self.syn_shenzhen.detach().numpy()
        self.shenzhen_label = np.zeros(len(self.syn_shenzhen)) + 7
        print(self.syn_shenzhen.shape)

        self.combined_train_data = np.concatenate((self.syn_running, self.syn_jumping), axis=0)
        self.combined_train_label = np.concatenate((self.running_label, self.jumping_label), axis=0)
        self.combined_train_label = self.combined_train_label.reshape(self.combined_train_label.shape[0], 1)

        if self.data_mode == 'JumpingRunning':
            print(self.combined_train_data.shape)
            print(self.combined_train_label.shape)
        else:
            print(self.syn_shenzhen.shape)
            print(self.shenzhen_label.shape)

    def __len__(self):
        if self.data_mode == 'JumpingRunning':
            return self.sample_size * 2
        else:
            return self.sample_size

    def __getitem__(self, idx):
        if self.data_mode == 'JumpingRunning':
            return self.combined_train_data[idx], self.combined_train_label[idx]
        else:
            return self.syn_shenzhen[idx], self.shenzhen_label[idx]


if __name__ == '__main__':
    syn_data = Synthetic_Dataset(data_mode='Shenzhen')
    print(syn_data)
