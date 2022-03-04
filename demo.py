import matplotlib.pyplot as plt
import torch
from torch import nn
from model import rho_CNN
import numpy as np
from dataloader import Demo_set,load
from tqdm import tqdm


import random

def load_model (model,path):

    check_point = torch.load(path)
    model.load_state_dict (check_point['model_state_dict'])


    model.eval()

    return model

def save_map (q_map,input_img):
    q_map = q_map.cpu().numpy()
    input_img = input_img.cpu().numpy()
    print(q_map.shape)
    q_map = q_map[0,0,:,:]
    input_img = input_img[0,0,:,:]
    plt.figure(1)
    plt.imshow(q_map,vmin=0.015,vmax=0.085,cmap='jet')
    plt.colorbar()
    plt.savefig('map/liver_trho.png')
    plt.figure(2)
    plt.imshow(input_img,cmap='gray')
    plt.savefig('map/liver_inputpng')
    
    
    
def testing (model,loader):

    print('Quantification starts')
    with torch.no_grad():
     for input_img in tqdm(loader):
   
        input_img = input_img.cuda()
        input_img = torch.unsqueeze(input_img,1)
        input_max = torch.amax(input_img, dim=(2, 3), keepdim=True)
        input_min = torch.amin(input_img, dim=(2, 3), keepdim=True)

        input_x = (input_img - input_min) / (input_max - input_min)



        input_img = input_img.float()

        output = model(input_x.float())
        t1rho = output[0]


        vis_mask = input_img.cpu().numpy()
        vis_mask[vis_mask>=1e-9]=1
        vis_mask[vis_mask<1e-9]=0
        vis_mask = torch.tensor(vis_mask).cuda()
        t1rho = t1rho*vis_mask

        save_map(t1rho,input_img)

    print('\nDone!')



def main(model,file_list):

    #dataset = Testing_pairs_set(file_list)
    dataset = Demo_set(file_list)
    loader = load(dataset, 1)
   
    testing(model,loader)
   

if __name__ == '__main__':
    image_file = 'liver.nii'
    model =rho_CNN()
    model = model.cuda()
    path = 'models/model.pt'
    model = load_model(model,path)
    main(model,[image_file])