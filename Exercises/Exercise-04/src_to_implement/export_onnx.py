import sys
import os
import socket
import torch                #type: ignore
import torchvision as tv    #type: ignore

from trainer import Trainer
from model import ResNet

hostname = socket.gethostname()

ROOT = '/home/jovyan/work/' if hostname == '7efc412e2797' else './'
dataset_path = os.path.join(ROOT, 'data.csv')

epoch = int(sys.argv[1])
model = ResNet()

criterion   = torch.nn.BCELoss()
trainer     = Trainer(model, criterion)
trainer.restore_checkpoint(epoch)
trainer.save_onnx(os.path.join(ROOT, 'checkpoint_{:03d}.onnx'.format(epoch)))
