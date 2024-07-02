import torch            #type: ignore
import numpy as np
import pandas as pd     #type: ignore
import os
import socket

from data       import ChallengeDataset
from trainer    import Trainer
from matplotlib import pyplot as plt
from model      import ResNet
from sklearn.model_selection import train_test_split


environment_type = os.getenv('ENVIRONMENT_TYPE')

ROOT = '/home/jovyan/work/exercise4_material/src_to_implement/' if environment_type == 'cuda-env' else './'

def get_dataloader(dataset_path):
    dataset = pd.read_csv(dataset_path, sep=';')
    train_data, val_data = train_test_split(dataset, test_size=0.2)

    train_dataloader = torch.utils.data.DataLoader(ChallengeDataset(train_data, 'train'), batch_size = 64, shuffle = True)
    val_dataloader   = torch.utils.data.DataLoader(ChallengeDataset(val_data, 'val'), batch_size = 64)

    return train_dataloader, val_dataloader

def run_training(epochs = 30):
    dataset_path = os.path.join(ROOT, 'data.csv')
    train_dataloader, val_dataloader = get_dataloader(dataset_path)
    model     = ResNet()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-5)
    trainer   = Trainer(model, criterion, optimizer, train_dataloader, val_dataloader, cuda = True)

    output_matrics = trainer.fit(epochs)
    return output_matrics

def plot_matrics(matrics):
    plt.plot(np.arange(len(matrics[0])), matrics[0], label = 'train loss')
    plt.plot(np.arange(len(matrics[1])), matrics[1], label = 'val loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(ROOT, 'losses.png'))


def main():
    output_matrics = run_training(epochs = 50)
    plot_matrics(output_matrics)
    
if __name__ == '__main__':
    main()