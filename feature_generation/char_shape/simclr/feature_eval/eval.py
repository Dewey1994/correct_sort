from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import torch
import torchvision
from PIL import Image
from feature_generation.char_shape.simclr.data_aug.contrastive_learning_dataset import *
from feature_generation.char_shape.simclr.models.resnet_simclr import ResNetSimCLR
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

img1 = Image.open('../characterImage/呆.png')
img2 = Image.open('../characterImage/保.png')


trans = ContrastiveLearningDataset.get_simclr_pipeline_transform_infer()

img1_tensor = trans(img1)
img2_tensor = trans(img2)

model = ResNetSimCLR(base_model='resnet18', out_dim=128)
modelsd = torch.load(r'C:\Users\40169\PycharmProjects\similarCharacterDy\SimCLR\runs\Jan13_00-39-51_DESKTOP-DEWEY\checkpoint_0050.pth.tar')
model.load_state_dict(modelsd['state_dict'])
model.eval()
cosine = torch.nn.CosineSimilarity()
with torch.no_grad():
    o1 = model(img1_tensor.unsqueeze(0))
    o2 = model(img2_tensor.unsqueeze(0))
    sim = cosine(o1,o2)
    print(sim.item())




