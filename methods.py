import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from encoder import ResNet_SSL, ResNet_SimSiam, MLPhead, LensNet
from encoder import BasicBlock, BottleNeck
from utils import schedule_byol_tau, concat_all_gather


def get_sscl_method(method_name="moco"):
    if method_name == 'moco':
        return MoCo
    elif method_name == 'simclr':
        return SimCLR
    elif method_name == 'byol':
        return BYOL
    elif method_name == 'simsiam':
        return SimSiam
    elif method_name == 'simttur':
        return SimTTUR
    elif method_name == 'disccon':
        return DiscCon
    elif method_name == 'moclr':
        return MoCLR


def EqCo(temperature, K, alpha):
    margin = temperature * np.log(alpha / K)
    
    return margin


class MoCo(nn.Module):
    def __init__(self, config):
        super(MoCo, self).__init__()
        self.config = config
        self.K = config.train.num_neg  
        self.tau = config.train.tau
        self.T = config.train.temperature
        
        # defining optimizing options : (1) Equivalent rule (2) debiasing (3) reweighting for hard negatives
        self.use_eqco = False
        self.margin = 0.
        if config.train.use_eqco:
            self.use_eqco = True
            self.alpha = self.K
            self.K = config.train.eqco_k
            self.margin = EqCo(self.T, self.K, self.alpha)
        
        self.use_dcl = False
        if config.train.use_dcl:
            self.use_dcl = True
            self.tau_plus = config.train.tau_plus
        
        self.use_hcl = False
        if config.train.use_hcl:
            self.use_hcl = True
            self.beta = config.train.beta
            
        self.ssl_feat_dim = config.model.ssl_feature_dim
        encoder_params = {'BN' : config.model.bn_encoder, 'norm_layer' : config.model.normalize, 'is_cifar' : 'cifar' in config.dataset.name}
        self.query_encoder = ResNet_SSL(config.model.arch, config.model.head, 
                                      encoder_params=encoder_params, ssl_feat_dim=self.ssl_feat_dim, bn_mlp=config.model.bn_proj)
        self.key_encoder = ResNet_SSL(config.model.arch, config.model.head, 
                                      encoder_params=encoder_params, ssl_feat_dim=self.ssl_feat_dim, bn_mlp=config.model.bn_proj)
        # this should be commented out
        # self.query_encoder = ResNet_SimSiam(arch_name=config.model.arch, encoder_params=encoder_params, ssl_feat_dim=2048, bn_mlp=config.model.bn_proj, regular_pred=False)
        # self.key_encoder = ResNet_SimSiam(arch_name=config.model.arch, encoder_params=encoder_params, ssl_feat_dim=2048, bn_mlp=config.model.bn_proj, regular_pred=False)
        
        self._init_encoders()
        self.register_buffer("neg_queue", torch.randn(self.ssl_feat_dim, self.K)) # [dim, K]
        # this should be commented out
        # self.register_buffer("neg_queue", torch.randn(2048, self.K))
        self.neg_queue = F.normalize(self.neg_queue, dim=0)
        self.register_buffer("queue_pointer", torch.zeros(1, dtype=torch.long))
    
    def _init_encoders(self):
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
    @torch.no_grad()
    def _update_key_encoder(self):
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data = self.tau * param_k.data + (1 - self.tau) * param_q.data
            
    @torch.no_grad()
    def _deque_and_enqueue(self, keys):
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        pointer = int(self.queue_pointer) # self.queue_pointer.item()
        
        assert self.K % batch_size == 0
        self.neg_queue[:, pointer: pointer + batch_size] = keys.t()
        pointer = (pointer + batch_size) % self.K
        
        self.queue_pointer[0] = pointer
    
    # For MoCo, negative queue's batch statistics are fixed
    # so that, for the same batch x_q and x_k, there would be 
    # larger similarity than negative queue, it leads to model to
    # learn only batch statistics difference from negatives (i.e. shortcut)
    # However, in case of SimCLR, negatives change across every training step
    # so that shuffling BN is not necessary
    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this
        idx_shuffle = torch.randperm(batch_size_all).cuda()
        dist.broadcast(idx_shuffle, src=0)
        idx_unshuffle = torch.argsort(idx_shuffle)
        gpu_idx = dist.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
        
        return x_gather[idx_this], idx_unshuffle
    
    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this
        gpu_idx = dist.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        
        return x_gather[idx_this]
        
    def forward(self, view_1, view_2):
        # already normalized
        q = self.query_encoder(view_1)
        
        with torch.no_grad():
            self._update_key_encoder()
            view_2, index_unshuffle = self._batch_shuffle_ddp(view_2)
            k = self.key_encoder(view_2) # already normalized
            k = self._batch_unshuffle_ddp(k, index_unshuffle)
        
        pos = torch.einsum('nd,nd->n', [q, k]).unsqueeze(-1)
        neg = torch.einsum('nd,dk->nk', [q, self.neg_queue.clone().detach()])
        
        pos_eqco = pos - self.margin # if self.use_eqco == False -> self.margin = 0
        if self.use_dcl:
            pos_exp = torch.exp(pos / self.T)
            neg_exp = torch.exp(neg / self.T)
            
            if self.use_hcl:
                importance = torch.exp(self.beta * neg / self.T)
                neg_exp = importance * neg_exp / importance.mean(dim=-1, keepdim=True)  
                          
            neg_exp = (-self.tau_plus * pos_exp + neg_exp) / (1 - self.tau_plus)
            neg_exp = torch.clamp(neg_exp, min=np.exp(-1 / self.T))
            
            pos_eqco_exp = torch.exp(pos_eqco / self.T)
            logits = torch.log(torch.cat([pos_eqco_exp, neg_exp], dim=1))
            
        else:
            logits = torch.cat([pos_eqco, neg], dim=1)
            logits = logits / self.T
            
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        logits_original = torch.cat([pos, neg], dim=1)
        
        self._deque_and_enqueue(k)
        
        return logits, labels, logits_original
            

class SimCLR(nn.Module):
    def __init__(self, config):
        super(SimCLR, self).__init__()
        self.config = config
        self.alpha = 4096
        self.K = 256
        self.T = config.train.temperature
        self.use_symmetric_logit = config.train.use_symmetric_logit
        
        self.use_eqco = False
        self.margin = 0.
        if config.train.use_eqco:
            self.use_eqco = True
            self.K = config.train.eqco_k
            self.margin = EqCo(self.T, self.K, self.alpha)
        
        self.use_dcl = False
        if config.train.use_dcl:
            self.use_dcl = True
            self.tau_plus = config.train.tau_plus
            
        self.use_hcl = False
        if config.train.use_hcl:
            self.use_hcl = True
            self.beta = config.train.beta
         
        self.ssl_feat_dim = config.model.ssl_feature_dim
        encoder_params = {'BN' : config.model.bn_encoder, 'norm_layer' : None, 'is_cifar' : 'cifar' in config.dataset.name}
        # self.encoder = ResNet_SSL(config.model.arch, config.model.head, 
        #                           encoder_params=encoder_params, ssl_feat_dim=self.ssl_feat_dim, bn_mlp=config.model.bn_proj)
        # this should be commented out 
        self.encoder = ResNet_SimSiam(arch_name=config.model.arch, encoder_params=encoder_params, ssl_feat_dim=2048, bn_mlp=config.model.bn_proj, regular_pred=False)
        
    def forward(self, view_1, view_2):
        batch_size = self.config.train.batch_size
        batch_size_this = view_1.shape[0]
        
        z_1 = self.encoder(view_1)
        z_2 = self.encoder(view_2)
        """
        (1) symmetric_logit 
            K : 2 * batch_size - 2 per 2 queries
            N : 2 * batch_size 
        (2) normal
            K : batch_size - 1 per 1 query
            N : batch_size
        """
        assert batch_size % batch_size_this == 0
        
        if self.use_symmetric_logit:
            features = torch.cat([z_1, z_2], dim=0) # [2 * N, 128] # N = 256 per each gpu
            dot_similarities = torch.mm(features, features.t()) # [2 * N, 2 * ]
            
            pos_ij = torch.diag(dot_similarities, batch_size_this)
            pos_ji = torch.diag(dot_similarities, -batch_size_this)
            pos = torch.cat([pos_ij, pos_ji]).view(2 * batch_size_this, -1)
            
            diagonal = np.eye(2 * batch_size_this)
            pos_ij_eye = np.eye(2 * batch_size_this, k=batch_size_this)
            pos_ji_eye = np.eye(2 * batch_size_this, k=-batch_size_this)

            neg_mask = torch.from_numpy(1 - (diagonal + pos_ij_eye + pos_ji_eye)).cuda().bool()
            neg = dot_similarities[neg_mask].view(2 * batch_size_this, -1)

            if self.K < 256:
                assert self.use_eqco
                selection_mask = torch.stack([torch.cat([torch.ones(2 * self.K), torch.zeros(neg.shape[1] - 2 *  self.K)])[torch.randperm(neg.shape[1])] 
                                              for _ in range(2 * batch_size_this)], dim=0).cuda().bool()
                neg = neg[selection_mask].view(2 * batch_size_this, -1)
        else:
            dot_similarities = torch.mm(z_1, z_2.t()) # [N, N]
            pos = torch.diag(dot_similarities).unsqueeze(-1) # [N, 1]
            
            diagonal = torch.eye(batch_size)
            neg_mask = (1 - diagonal).cuda().bool()
            neg = dot_similarities[neg_mask].view(batch_size, -1) # [N, N - 1]

            if self.K < 256:
                one_zeros = torch.cat([torch.ones(self.K), torch.zeros(neg.shape[1] - self.K)])
                selection_mask = torch.stack([one_zeros[torch.randperm(neg.shape[1])] for _ in range(batch_size)], dim=0)
                selection_mask = selection_mask.cuda().bool()
                neg = neg[selection_mask].view(2 * batch_size, -1)
        
        pos_eqco = pos - self.margin
        if self.use_dcl:
            pos_exp = torch.exp(pos / self.T)
            neg_exp = torch.exp(neg / self.T)
            
            if self.use_hcl:
                importance = torch.exp(self.beta * neg / self.T)
                neg_exp = importance * neg_exp / importance.mean(dim=-1, keepdim=True)
            
            neg_exp = (-self.tau_plus * pos_exp + neg_exp) / (1 - self.tau_plus)
            neg_exp = torch.clamp(neg_exp, min=np.exp(-1 / self.T))
            
            pos_eqco_exp = torch.exp(pos_eqco / self.T)
            logits = torch.log(torch.cat([pos_eqco_exp, neg_exp], dim=1))
        else:
            logits = torch.cat([pos_eqco, neg], dim=1)
            logits = logits / self.T
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        # this should be commented out
        # labels = torch.arange(logits.shape[1], dtype=torch.long).repeat(logits.shape[0], 1).cuda()
        logits_original = torch.cat([pos, neg], dim=1)
        
        return logits, labels, logits_original
    

class SimCLR_OOD(SimCLR):
    def __init__(self, config):
        super(SimCLR_OOD, self).__init__(config)
        self.config = config
        
    def forward(self, view_1, view_2, ood_view):
        raise NotImplementedError
    
    
        
class BYOL(nn.Module):
    def __init__(self, config):
        super(BYOL, self).__init__()
        self.config = config
        
        self.tau = config.train.tau
        if config.model.ssl_feature_dim == 128:
            # self.ssl_feat_dim = 2 * config.model.ssl_feature_dim
            self.ssl_feat_dim = config.model.ssl_feature_dim
        else:
            self.ssl_feat_dim = config.model.ssl_feature_dim
            
        encoder_params = {'BN' : config.model.bn_encoder, 'norm_layer' : config.model.normalize, 'is_cifar' : 'cifar' in config.dataset.name}
        # self.online_network = ResNet_SSL(config.model.arch, config.model.head, 
        #                                  encoder_params=encoder_params, 
        #                                  ssl_feat_dim=self.ssl_feat_dim, hidden_double=True, bn_mlp=config.model.bn_proj)
        # self.target_network = ResNet_SSL(config.model.arch, config.model.head, 
        #                                  encoder_params=encoder_params,
        #                                  ssl_feat_dim=self.ssl_feat_dim, hidden_double=True, bn_mlp=config.model.bn_proj)
        
        # this should be commented out
        self.online_network = ResNet_SimSiam(arch_name=config.model.arch, encoder_params=encoder_params, ssl_feat_dim=2048, bn_mlp=config.model.bn_proj, 
                                             regular_pred=False)
        self.target_network = ResNet_SimSiam(arch_name=config.model.arch, encoder_params=encoder_params, ssl_feat_dim=2048, bn_mlp=config.model.bn_proj, 
                                             regular_pred=False)
        
        self._init_encoders()
        
        hidden = self.online_network.proj_head.fc1.out_features
        # this should be commented out
        self.predictor = MLPhead(2048, 2048, 512, bn_mlp=config.model.bn_pred)
        # self.predictor = MLPhead(self.ssl_feat_dim, self.ssl_feat_dim, hidden=hidden, bn_mlp=config.model.bn_pred)
    
    def _init_encoders(self):
        for param_online, param_target in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_target.data.copy_(param_online.data)
            param_target.requires_grad = False
                
    @torch.no_grad()
    def _update_target_network(self):
        for param_online, param_target in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_target.data = self.tau * param_target.data + (1 - self.tau) * param_online.data

    def forward(self, view_1, view_2):
        z_1_online = self.online_network(view_1)
        z_2_pred = self.predictor(z_1_online)
        z_2_pred = F.normalize(z_2_pred, dim=1)
        z_2_target = self.target_network(view_2)
        loss_1 = 2 - 2 * (z_2_pred * z_2_target).sum(dim=-1)
        
        z_2_online = self.online_network(view_2)
        z_1_pred = self.predictor(z_2_online)
        z_1_pred = F.normalize(z_1_pred, dim=1)
        z_1_target = self.target_network(view_1)
        loss_2 = 2 - 2 * (z_1_pred * z_1_target).sum(dim=-1)
        
        self._update_target_network()
        
        return loss_1 + loss_2


class SimSiam(nn.Module):
    def __init__(self, config):
        super(SimSiam, self).__init__()
        self.config = config
        self.ssl_feat_dim = config.model.ssl_feature_dim
            
        encoder_params = {'BN' : config.model.bn_encoder, 'norm_layer' : None, 'is_cifar' : 'cifar' in config.dataset.name}
        self.encoder = ResNet_SimSiam(arch_name=config.model.arch, encoder_params=encoder_params, ssl_feat_dim=self.ssl_feat_dim, bn_mlp=config.model.bn_proj, regular_pred=config.model.regular_pred)
        
        if config.model.regular_pred:
            model_dict = {
                'resnet18' : 512,
                'resnet50' : 2048
            }
            hidden = model_dict[config.model.arch]
            self.predictor = MLPhead(in_features=self.ssl_feat_dim, hidden=hidden, out_features=self.ssl_feat_dim, bn_mlp=config.model.bn_pred)
        else:
            self.predictor = MLPhead(in_features=self.ssl_feat_dim, hidden=int(self.ssl_feat_dim / 4), out_features=self.ssl_feat_dim, bn_mlp=config.model.bn_pred)
        
    def forward(self, view_1, view_2):
        z_1 = self.encoder(view_1)
        z_2 = self.encoder(view_2)
        
        z_2_pred = self.predictor(z_1)
        z_2_pred = F.normalize(z_2_pred, dim=1)
        z_1_pred = self.predictor(z_2)
        z_1_pred = F.normalize(z_1_pred, dim=1)
        
        loss = -0.5 * ((z_1_pred * z_1.detach()).sum(dim=-1) + (z_2_pred * z_2.detach()).sum(dim=-1))
        
        return loss


class HSA(nn.Module):
    def __init__(self, config):
        self.config = config
        super(HSA, self).__init__()
    
    def forward(self, view_1, view_2):
        raise NotImplementedError
    
    
class CAST(nn.Module):
    def __init__(self, config):
        self.config = config
        super(CAST, self).__init__()
    
    def forward(self, view_1, view_2):
        raise NotImplementedError


#####################################################################################################################################################
###################################################################### My Idea ######################################################################
#####################################################################################################################################################


class DiscCon(SimCLR):
    def __init__(self, config):
        super(DiscCon, self).__init__(config)
        self.discriminator = nn.Sequential(
            nn.Linear(2 * self.ssl_feat_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )
        
    def forward(self, view_1, view_2, use_D=False):
        batch_size = view_1.shape[0]
        
        z_1 = self.encoder(view_1)
        z_2 = self.encoder(view_2)
        features = torch.cat([z_1, z_2], dim=0)
        
        sim = torch.mm(features, features.t())
        pos_ij = torch.diag(sim, batch_size)
        pos_ji = torch.diag(sim, -batch_size)
        pos = torch.cat([pos_ij, pos_ji], dim=0).view(2 * batch_size, -1)
        diagonal = np.eye(2 * batch_size)
        pos_ij_eye = np.eye(2 * batch_size, k=batch_size)
        pos_ji_eye = np.eye(2 * batch_size, k=-batch_size)
        
        neg_mask = torch.from_numpy(1 - (diagonal + pos_ij_eye + pos_ji_eye)).cuda().bool()
        neg = sim[neg_mask].view(2 * batch_size, -1)
        
        logits = torch.cat([pos, neg], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        if use_D:
            d_input_pos_ij = torch.cat([z_1, z_2], dim=1)
            # d_input_pos_ji = torch.cat([z_2, z_1], dim=1)
            
            # mask_ij = torch.from_numpy(1 - (diagonal + pos_ij_eye)[:batch_size, :]).long().cuda() 
            # mask_ji = torch.from_numpy(1- (diagonal + pos_ji_eye)[batch_size:, :]).long().cuda()
            mask_ij = torch.from_numpy(1 - pos_ij_eye)[:batch_size, batch_size:].long().cuda() 
            # mask_ji = torch.from_numpy(1 - pos_ji_eye)[batch_size:, :batch_size].long().cuda()
            _, hardest_indices_ij = (sim[:batch_size, batch_size:] * mask_ij).max(dim=1)
            # _, hardest_indices_ji = (sim[batch_size:, :batch_size] * mask_ji).max(dim=1)
            
            d_input_neg_ij = torch.cat([z_1, z_2[hardest_indices_ij]], dim=1)
            # d_input_neg_ji = torch.cat([z_2, z_1[hardest_indices_ji]], dim=1)
            
            # d_logit_pos = self.discriminator(torch.cat([d_input_pos_ij, d_input_pos_ji], dim=0))
            # d_logit_neg = self.discriminator(torch.cat([d_input_neg_ij, d_input_neg_ji], dim=0))
            d_logit_pos = self.discriminator(d_input_pos_ij)
            d_logit_neg = self.discriminator(d_input_neg_ij)
        
            return logits, labels, (d_logit_pos, d_logit_neg)
        else:
            return logits, labels        
 
 
class MoCLR(nn.Module):
    def __init__(self, config):
        super(MoCLR, self).__init__()
        self.config = config
        self.K = config.train.num_neg
        self.tau = config.train.tau
        self.T = config.train.temperature
        # self.pos_momentum_full = config.train.pos_momentum_full
        
        self.ssl_feat_dim = config.model.ssl_feature_dim
        encoder_params = {'BN' : config.model.bn_encoder, 
                          'norm_layer' : None,
                          'is_cifar' : 'cifar' in config.dataset.name}
        
        self.encoder = ResNet_SSL(arch_name=config.model.arch, encoder_params=encoder_params, 
                                  ssl_feat_dim=self.ssl_feat_dim, bn_mlp=config.model.bn_proj)
        self.momentum_encoder = ResNet_SSL(arch_name=config.model.arch, encoder_params=encoder_params, 
                                           ssl_feat_dim=self.ssl_feat_dim, bn_mlp=config.model.bn_proj)

        self.option = config.train.moclr_option
        if (self.option == 'hybrid') or (self.option == 'siamese'):
            hidden = self.encoder.proj_head.fc1.out_features
            self.predictor = MLPhead(self.ssl_feat_dim, self.ssl_feat_dim, hidden=hidden, bn_mlp=config.model.bn_pred)
        
        self.pos_momentum_full = config.train.pos_momentum_full
        
    def _init_encoders(self):
        for param_encoder, param_momentum in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_momentum.data.copy_(param_encoder)
            param_momentum.requires_grad = False
            
    
    @torch.no_grad()
    def _update_momentum_encoder(self):
        for param_encoder, param_momentum in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_momentum.data = self.tau * param_momentum.data + (1 - self.tau) * param_encoder.data

    def forward(self, view_1, view_2, use_momentum=True):
        z_1 = self.encoder(view_1)
        z_2 = self.encoder(view_2)
        
        z_1_momentum = self.momentum_encoder(view_1)
        z_2_momentum = self.momentum_encoder(view_2)
        
        batch_size = z_1.shape[0]
        neg_mask = torch.from_numpy(1 - (np.eye(2 * batch_size) + np.eye(2 * batch_size, k=batch_size) + np.eye(2 * batch_size, k=-batch_size))).bool().cuda()
             
        features = torch.cat([z_1, z_2], dim=0)
        sim = torch.mm(features, features.t()) / self.T
        neg = sim[neg_mask].view(2 * batch_size, -1)
        
        # another option : only add z_1 * z_1_momentum or z_1 * z_2_momentum
        # another option : how about remaining negatives encoded from momentum?     
        # or use additional negatives using momentum encoded representation z_momentum?
        # consideratoin : should detach something?   
        
        pos_base = torch.cat([torch.diag(sim, batch_size), torch.diag(sim, -batch_size)], dim=0)  
        logits_original = torch.cat([pos_base.unsqueeze(-1), neg], dim=1)
        labels = torch.zeros(pos_base.shape[0], dtype=torch.long).cuda() 
        
        if use_momentum:
            if self.option == 'contrast':       
                if self.pos_momentum_full: 
                    pos_ij = ((z_1 * z_2_momentum).sum(dim=1) + (z_1 * z_1_momentum).sum(dim=1)) / self.T
                    pos_ji = ((z_2 * z_1_momentum).sum(dim=1) + (z_2 * z_2_momentum).sum(dim=1)) / self.T
                else:
                    pos_ij = (z_1 * z_2_momentum).sum(dim=1) / self.T
                    pos_ji = (z_2 * z_1_momentum).sum(dim=1) / self.T
                pos = torch.cat([pos_ij, pos_ji], dim=0) + pos_base
                pos = pos / (3 if self.pos_momentum_full else 2)
                
                # pos_exp = torch.exp(pos)
                # neg_exp = torch.exp(neg).sum(dim=1)
                # loss = -torch.log(pos_exp / (pos_exp + neg_exp))    
                logits = torch.cat([pos.unsqueeze(-1), neg], dim=1)
                
                # self._update_momentum_encoder()
                
                # return loss, labels, logits_original
                return logits, labels, logits_original        
            
            elif self.option == 'hybrid':
                z_2_pred = F.normalize(self.predictor(z_1), dim=1)
                z_1_pred = F.normalize(self.predictor(z_2), dim=1)
                
                loss_siam = -0.5 * ((z_1_pred * z_1.detach()).sum(dim=1) + (z_2_pred * z_2.detach()).sum(dim=1))
                
                features_12 = torch.cat([z_1, z_2_momentum], dim=0)
                sim_12 = torch.mm(features_12, features_12.t())
                neg_12 = sim_12[neg_mask].view(2 * batch_size, -1)
                pos_12 = torch.cat([torch.diag(sim_12, batch_size), torch.diag(sim_12, -batch_size)], dim=0)
                logits_12 = torch.cat([pos_12.unsqueeze(-1), neg_12], dim=1)
                
                features_21 = torch.cat([z_2, z_1_momentum], dim=0)
                sim_21 = torch.mm(features_21, features_21.t())
                neg_21 = sim_21[neg_mask].view(2 * batch_size, -1)
                pos_21 = torch.cat([torch.diag(sim_21, batch_size), torch.diag(sim_21, -batch_size)], dim=0)
                logits_21 = torch.cat([pos_21.unsqueeze(-1), neg_21], dim=1)
                
                labels = torch.zeros(pos_12.shape[0], dtype=torch.long).cuda()
                
                self._update_momentum_encoder()
                
                return loss_siam, logits_12, logits_21, labels, logits_original
            
            elif self.option == 'siamese':
                z_1_momentum_pred = F.normalize(self.predictor(z_2), dim=1)
                z_1_pred = F.normalize(self.predictor(z_2), dim=1)
                z_2_momentum_pred = F.normalize(self.predictor(z_1), dim=1)
                z_2_pred = F.normalize(self.predictor(z_1), dim=1)
                
                loss_12 = (2 - 2 * (z_1_momentum_pred * z_1_momentum).sum(dim=1)) + (2 - 2 * (z_1_pred * z_1.detach()).sum(dim=1)) / 2
                loss_21 = (2 - 2 * (z_2_momentum_pred * z_2_momentum).sum(dim=1)) + (2 - 2 * (z_2_pred * z_2.detach()).sum(dim=1)) / 2
                
                self._update_momentum_encoder()
                
                return loss_12 + loss_21
        else:
            return logits_original, labels
        

class SemanticCon(nn.Module):
    def __init__(self, config):
        super(SemanticCon, self).__init__()
        self.config = config

    def forward(self, view_1, view_2):
        raise NotImplementedError
    
        
    
class PseudoCLR(nn.Module):
    def __init__(self, config):
        super(PseudoCLR, self).__init__()
        self.config = config
        self.K = config.train.num_neg
        self.T = config.train.temperature
        self.C = config.train.num_pseudo_class
        self.ssl_feat_dim = config.train.ssl_feature_dim
        
        encoder_params = {'BN' : True, 'norm_layer' : None, 'is_cifar' : 'cifar' in config.dataset.name}
        self.encoder = ResNet_SSL(arch_name=config.model.arch, encoder_params=encoder_params, ssl_feat_dim=self.ssl_feat_dim, bn_mlp=config.model.bn_proj)
         
    def forward(self, x):
        raise NotImplementedError
    
    def cluster_minibatch(self, z):
        raise NotImplementedError
    
    def find_near_center(self):
        raise NotImplementedError
    
    
class LensCon_MoCo(MoCo):
    def __init__(self, config):
        super(LensCon_MoCo, self).__init__(config)
        self.lens = LensNet(BasicBlock)
        
    def forward(self, view_1, view_2):
        view_adv_1 = self.lens(view_1)
        view_adv_2 = self.lens(view_2)
        
    
    
class SimTTUR(nn.Module):
    def __init__(self, config):
        super(SimTTUR, self).__init__()
        self.config = config 
        self.ssl_feat_dim = config.model.ssl_feature_dim
        
        encoder_params = {'BN' : config.model.bn_encoder, 'norm_layer' : None, 'is_cifar' : 'cifar' in config.dataset.name}
        self.encoder_fast = ResNet_SSL(config.model.arch, config.model.head, encoder_params=encoder_params, ssl_feat_dim=self.ssl_feat_dim, bn_mlp=config.model.bn_proj)
        self.encoder_slow = ResNet_SSL(config.model.arch, config.model.head, encoder_params=encoder_params, ssl_feat_dim=self.ssl_feat_dim, bn_mlp=config.model.bn_proj)
        
        feat = 512 if config.model.arch == 'resnet18' else 2048
        self.predictor = MLPhead(in_features=self.ssl_feat_dim, hidden=feat, out_features=self.ssl_feat_dim, bn_mlp=config.model.bn_pred)
        
    def forward(self, view_1, view_2):
        z_1_fast = self.encoder_fast(view_1)
        z_2_pred_fast = F.normalize(self.predictor(z_1_fast), dim=1)
        z_2_slow = self.encoder_slow(view_2)
        z_1_pred_slow = F.normalize(self.predictor(z_2_slow), dim=1)
        
        z_2_fast = self.encoder_fast(view_2)
        z_1_pred_fast = F.normalize(self.predictor(z_2_fast), dim=1)
        z_1_slow = self.encoder_slow(view_1)
        z_2_pred_slow = F.normalize(self.predictor(z_1_slow), dim=1)
        
        loss = -0.25 * ((z_2_pred_fast * z_2_slow.detach()).sum(dim=-1) + (z_1_pred_fast * z_1_slow.detach()).sum(dim=-1) 
                        + (z_2_pred_slow * z_2_fast.detach()).sum(dim=-1) + (z_1_pred_slow * z_2_fast.detach()).sum(dim=-1))
        
        return loss
        
        
        