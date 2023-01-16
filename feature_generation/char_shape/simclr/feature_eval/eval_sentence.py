from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import torch
import torchvision
from PIL import Image
from data_aug.contrastive_learning_dataset import *
from models.resnet_simclr import ResNetSimCLR
import os
import sys


sys.path.append("..")


def get_char_dict():
    char_dict = dict()
    t = os.listdir('../characterImage')
    main_path = r"C:\Users\40169\PycharmProjects\SimCLR\characterImage"
    for p in t:
        key = p[0]
        val = os.path.join(main_path,p)
        char_dict[key]=val
    return char_dict


char_dict = get_char_dict()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)


strs_list = ['招聘保安五险一金包吃住','招聘呆安五捡一金包吃住','招聘保女五险一金包吃住']
img_batches = []
trans = ContrastiveLearningDataset.get_simclr_pipeline_transform_infer()

for strs in strs_list:
    imgs = []
    for str in strs:
        img = Image.open(char_dict[str])
        img = trans(img)
        imgs.append(img)
        imgs_tensor = torch.stack(imgs).view(-1,96,96)
    img_batches.append(imgs_tensor)


img_tensor = torch.stack(img_batches, dim=0)


model = ResNetSimCLR(base_model='resnet18', out_dim=128)
modelsd = torch.load(r'C:\Users\40169\PycharmProjects\similarCharacterDy\SimCLR\runs\Jan13_00-39-51_DESKTOP-DEWEY\checkpoint_0050.pth.tar')
model.load_state_dict(modelsd['state_dict'])
model.eval()
cosine = torch.nn.CosineSimilarity(dim=1)
with torch.no_grad():
    o1 = model(img_tensor)
   
    # print(sim[sim!=1].mean().item())




