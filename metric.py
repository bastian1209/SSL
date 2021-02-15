import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from methods import get_sscl_method
from configs.config import load_config_from_yaml
from data import get_dataset, get_loader
from glob import glob
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
from tqdm import tqdm


def measure_uniformity(features, intra_only=True, unif_t=3):
    if intra_only:
        assert type(features) != list
        z = features
    else:
        assert type(features) == list
        z, z_key = features
        batch_size = z.shape[0]
        similarity = torch.mm(z, z_key.t())
        pos = torch.diag(similarity).unsqueeze(-1) # [N, 1]
        neg =  similarity[(1 - torch.eye(batch_size)).bool()].view(batch_size, -1) # [N, N-1]
         
    pair_dist = torch.pdist(z, p=2).pow(2)
    if not intra_only:
        pair_dist_intra = torch.cat([pair_dist, torch.pdist(z_key, p=2).pow(2)])
        pair_dist_inter = (2 - 2 * neg).flatten()
        pair_dist = torch.cat([pair_dist_inter, pair_dist_intra])
    
    g_potential = pair_dist.mul(-unif_t).exp().mean().log()
    
    return g_potential


def measure_tolerance(z_1, z_2):
    tolerance = (z_1 * z_2).sum(dim=1)
    
    return tolerance.mean()


def measure_lin_sep():
    raise NotImplementedError


def cluster_energy(clusters, gt_labels, cluster_ids):
    centers = []
    
    # intra-clsuter energy
    intra_dist = 0.
    for index in cluster_ids:
        mask = gt_labels == index
        cluster = clusters[mask]
        center = cluster.mean(dim=0, keepdim=True)
        centers.append(center)
        intra_dist += ((cluster - center) ** 2).sum().sqrt().mean().cpu().item()
    intra_dist /= len(cluster_ids)
    
    # inter-cluster energy
    from itertools import combinations
    pairs = list(combinations(centers, 2))
    inter_dist = np.mean(list(map(lambda x : np.sqrt(np.sum(((x[0].cpu().numpy() - x[1].cpu().numpy()) ** 2))), pairs)))
 
    
    energy = intra_dist + inter_dist
    
    return energy, intra_dist, inter_dist


def to_single(ckpt):
    new = OrderedDict()
    for k, v in ckpt.items():
        prefix = 'module.'
        key = k.replace(prefix, '')
        new[key] = v
    
    return new


def prepare_model_data(method, model_path, mode='train', only_final=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config_path = model_path + 'experiment_config.yaml'
    config = load_config_from_yaml(config_path)
    if not only_final:
        model_ckpts = sorted(glob(os.path.join(model_path, "*0*0.pth.tar")))
    else:
        model_ckpts = sorted(glob(os.path.join(model_path, "*final.pth.tar")))
        
    model = get_sscl_method(method)(config)
    model.eval()
    
    # single_data_loader = get_loader(config, get_dataset(config, mode, multiview=False))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    single_dataset = datasets.CIFAR10(root='./dataset', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.Resize(40),
                                    transforms.CenterCrop(32),
                                    transforms.ToTensor(),
                                    normalize,
                                ]))
    single_data_loader = DataLoader(single_dataset, batch_size=config.train.batch_size, shuffle=True)
    multi_data_loader = get_loader(config, get_dataset(config, mode, multiview=True))
    
    return model, model_ckpts, single_data_loader, multi_data_loader, config
    
 
def pairwise_similarity(features):
    if type(features) != list:
        sim = torch.mm(features, features.t())
        mask = torch.eye(features.shape[0]).cuda()
    else:
        sim = torch.mm(features[0], features[1].t())    
        mask = torch.eye(features[0].shape[0]).cuda()

    sim = (1 - mask) * sim

    return sim.sum() 

 
    
def inspect_pos_sim(model, model_ckpts, multi_loader, config, use_momentum=False):
    class_dict_epoch = []
    
    encoder_name = {'moco' : 'query_encoder', 'byol' : 'online_network', 'simclr' : 'encoder', 'simsiam' : 'encoder', 'moclr' : 'encoder'}
    target_name = {'moco' : 'key_encoder', 'byol' : 'target_network', 'moclr' : 'momentum_encoder'} 
    encoder = encoder_name[config.method]
    if use_momentum:
        target = target_name[config.method]
    
    model.cuda()
    model.eval()
    for ckpt in tqdm(model_ckpts):
        class_dict = {}
        for i in range(10):
            class_dict[i] = []
        
        ckpt  = to_single(torch.load(ckpt)['model'])
        model.load_state_dict(ckpt)
        with torch.no_grad():
            for img, label in multi_loader:
                view_1 = img[0].cuda()
                view_2 = img[1].cuda()
                label = label.cuda()

                z_1 = model.__dict__['_modules'][encoder](view_1)
                if use_momentum:
                    z_2 = model.__dict__['_modules'][target](view_2)
                else:
                    z_2 = model.__dict__['_modules'][encoder](view_2)
                sim = (z_1 * z_2).sum(dim=1)
                
                for i in range(10):
                    class_dict[i].append(sim[label == i])
            for i in range(10):
                class_dict[i] = torch.cat(class_dict[i], dim=0)
        class_dict_epoch.append(class_dict)

    for cdict in class_dict_epoch:
        for i in range(10):
            cdict[i] = torch.mean(cdict[i])

    class_sim = torch.cat(list(map(lambda x : torch.stack(list(x.values()), dim=0).unsqueeze(1), class_dict_epoch)), dim=1)
    
    return class_sim


def inspect_in_class_sim(model, model_ckpts, single_loader, config, use_momentum=False):
    class_dict_intra_q_epoch = []
    if (config.method == 'moco') or (config.method == 'byol'):
        class_dict_intra_k_epoch = []
        class_dict_inter_epoch = []
    
    encoder_name = {'moco' : 'query_encoder', 'byol' : 'online_network', 'simclr' : 'encoder', 'simsiam' : 'encoder', 'moclr' : 'encoder'}
    target_name = {'moco' : 'key_encoder', 'byol' : 'target_network', 'moclr' : 'momentum_encoder'}
    encoder = encoder_name[config.method]
    if (config.method == 'moco') or (config.method == 'byol'):
        target = target_name[config.method]
    
    model.cuda()
    model.eval()
    for ckpt in tqdm(model_ckpts):
        class_dict_intra_q = {}
        class_dict_intra_k = {}
        class_dict_inter = {}
        for i in range(10):
            class_dict_intra_q[i] = 0.
            class_dict_intra_k[i] = 0.
            class_dict_inter[i] = 0.

        ckpt  = to_single(torch.load(ckpt)['model'])
        model.load_state_dict(ckpt)
        with torch.no_grad():
            class_nums = [0. for _ in range(10)]
            for img, label in single_loader:
                img = img.cuda()
                label = label.cuda()

                z = model.__dict__['_modules'][encoder](img) 
                if use_momentum:
                    z_momentum = model.__dict__['_modules'][target](img)
                
                for i in range(10):
                    mask = label == i
                    z_masked = z[mask]
                    if use_momentum:
                        z_momentum_masked = z_momentum[mask]
                    num_data = z_masked.shape[0]

                    class_dict_intra_q[i] += pairwise_similarity(z_masked).cpu()
                    if use_momentum:
                        class_dict_intra_k[i] += pairwise_similarity(z_momentum_masked).cpu()
                        class_dict_inter[i] += pairwise_similarity([z_masked, z_momentum_masked]).cpu()
                    class_nums[i] += (num_data ** 2 - num_data)
            
            for i in range(10):
                class_dict_intra_q[i] /= class_nums[i]
                if use_momentum:
                    class_dict_intra_k[i] /= class_nums[i]
                    class_dict_inter[i] /= class_nums[i]
                    
        class_dict_intra_q_epoch.append(class_dict_intra_q)
        if use_momentum:
            class_dict_intra_k_epoch.append(class_dict_intra_k)
            class_dict_inter_epoch.append(class_dict_inter)
        
    sim_q = torch.cat(list(map(lambda x : torch.stack(list(x.values()), dim=0).unsqueeze(1), class_dict_intra_q_epoch)), dim=1)
    if use_momentum:
        sim_k = torch.cat(list(map(lambda x : torch.stack(list(x.values()), dim=0).unsqueeze(1), class_dict_intra_k_epoch)), dim=1)        
        sim_inter = torch.cat(list(map(lambda x : torch.stack(list(x.values()), dim=0).unsqueeze(1), class_dict_inter_epoch)), dim=1)
    
        return sim_q, sim_k, sim_inter
    
    return sim_q
    