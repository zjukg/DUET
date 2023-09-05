import scipy.io as sio
from transformers import DeiTFeatureExtractor, DeiTModel, DeiTForImageClassification, DeiTConfig, SwinForImageClassification, AutoFeatureExtractor
from tqdm import tqdm
from PIL import Image
from torch.autograd import Variable
import torch
import pickle
import numpy as np

# cub
matcontent = sio.loadmat('/home/hyf/code/APN-ZSL/data/CUB/APN.mat')
image_files = matcontent['image_files']

# feature_extractor = AutoFeatureExtractor.from_pretrained("/data/hyf/data/PLMs/swin")
# swin = SwinForImageClassification.from_pretrained("/data/hyf/data/PLMs/swin").cuda()
feature_extractor = AutoFeatureExtractor.from_pretrained("/home/hyf/data/PLMs/deit-base-distilled-patch16-224")
swin = DeiTForImageClassification.from_pretrained("/home/hyf/data/PLMs/deit-base-distilled-patch16-224").cuda()


for i, image_path in enumerate(tqdm(image_files, total=len(image_files))):
    image_path = image_path[0][0].replace("/BS/Deep_Fragments/work/MSc", "/home/hyf/data")
    img = Image.open(image_path).convert('RGB')
    feature = feature_extractor(images=img, return_tensors="pt")['pixel_values']
    embedding = swin(feature.cuda()).logits.detach().cpu().numpy()
    if i == 0:
        image_embedding = embedding
    else:
        image_embedding = np.concatenate((image_embedding, embedding), 0)

with open("cub_image2embedding.pkl", "wb") as f:
    pickle.dump(image_embedding,f)