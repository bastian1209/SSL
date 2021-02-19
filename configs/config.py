from yacs.config import CfgNode as _CfgNode
from contextlib import redirect_stdout
import os
from utils import get_resume_info
from collections import OrderedDict
import yaml


def represent_dictionary_order(self, dict_data):
    return self.represent_mapping('tag:yaml.org,2002:map', dict_data.items())


def setup_yaml():
    yaml.add_representer(OrderedDict, represent_dictionary_order)


setup_yaml()    
class CfgNode(_CfgNode):
    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        super(CfgNode, self).__init__(init_dict, key_list, new_allowed)
    
    def dump(self, **kwargs):
        def convert_to_dict(cfg_node, key_list):
            if not isinstance(cfg_node, CfgNode):
                return cfg_node
            else:
                cfg_dict = OrderedDict(cfg_node)
                for k, v in cfg_dict.items():
                    cfg_dict[k] = convert_to_dict(v, key_list + [k])
                return cfg_dict
        
        self_as_dict = convert_to_dict(self, [])
        return yaml.dump(self_as_dict, **kwargs)
        
        
config = CfgNode()

config.system = CfgNode()
config.system.gpu = 0
config.system.num_workers = 16
config.system.distributed = True
config.system.dist_url = 'tcp://localhost:10001'
config.system.dist_backend = 'nccl'
config.system.multiprocessing_distributed = True
config.system.world_size = -1
config.system.rank = -1
config.system.print_freq = 10
config.system.save_period = 10
config.system.save_dir = './save'
config.system.log_dir = './log'

config.dataset = CfgNode()
config.dataset.root = './dataset'
config.dataset.name = 'ImageNet_100'
config.dataset.img_size = [224, 224, 3]
config.dataset.aug = 'simclr_aug'
config.dataset.num_classes = 100

config.train = CfgNode()
config.train.start_epoch = 0
config.train.epochs = 200
config.train.batch_size = 128
config.train.num_neg = 16384
config.train.num_view = 2
config.train.optim = 'SGD'
config.train.use_schedule = True
config.train.schedule = 'cos'
config.train.milestones = [120, 160]
config.train.base_lr = 0.03
config.train.momentum = 0.9
config.train.wd = 1e-4
config.train.temperature = 0.2 # MoCo v2 0.2, MoCo v1 0.07, SimCLR 0.5
config.train.tau = 0.999
config.train.use_eqco = False
config.train.eqco_k = None
config.train.use_dcl = False
config.train.tau_plus = 0.01
config.train.use_hcl = False
config.train.alpha = None # EqCo hparam
config.train.beta = 1.0 # HCL hparam
config.train.resume = None
config.train.use_symmetric_logit = True
config.train.use_temp_schedule = False
config.train.temp_warmup = 100
config.train.temp_schedule = 'linear'
config.train.unif_t = 3
config.train.use_wass = False
config.train.use_neg_wass = False
config.train.moclr_option = 'contrast'
config.train.pos_momentum_full = False
config.train.negcl_option = 'clhead' # or 'faiss'
config.train.num_cluster = 10
# config.train.num_pseudo_class = 10
# config.train.beta = 2z

config.model = CfgNode()
config.model.arch = 'resnet18'
config.model.normalize = None
config.model.head = 'mlp'
config.model.regular_pred = False
config.model.ssl_feature_dim = 128
config.model.bn_encoder = True
config.model.bn_proj = [False, False]
config.model.bn_pred = [False, False]


config.method = 'moco'


def inspect_config(config):
    if 'ImageNet' in config.dataset.name:
        assert config.dataset.img_size == [224, 224, 3]
        if config.dataset.name.endswith('100'):
            assert config.dataset.num_classes == 100
        elif config.dataset.name.endswith('1K'):
            assert config.dataset.num_classes == 1000
    elif 'cifar' in config.dataset.name:
        assert config.dataset.img_size == [32, 32, 3]
        if config.dataset.name.endswith('10'):
            assert config.dataset.num_classes == 10
        elif config.dataset.name.endswith('100'):
            assert config.dataset.num_classes == 100
    elif 'stl' in config.dataset.name:
        assert config.dataset.img_size == [96, 96, 3]
        assert config.dataset.num_classes == 10
    
    # if config.method == 'moco':
    #     assert config.model.bn_proj == [False, False]
    # elif config.method == 'simclr':
    #     assert config.model.bn_proj == [True, True] # need to be fixed to use SyncBN for BathcNorm1D
    # elif config.method == 'byol':
    #     pass
    
    print('valid configuration')
    

def get_config(method='moco'):
    # return base config for moco
    if method == 'moco':
        config.bn_proj = [False, False]
        
    elif method == 'simclr':
        config.method = 'simclr'
        config.train.temperature = 0.5 # 0.1 for large batch size
        config.train.base_lr = 0.1
        config.train.wd = 1e-6
        config.train.use_symmetric_logit = False
        config.bn_proj = [True, True]
    
    elif method == 'byol':
        config.train.tau = 0.996
        config.train.base_lr = 0.2
        config.train.wd = 1.5e-6
        config.model.ssl_feature_dim = 128
        
    elif method == 'simsiam':
        config.train.base_lr = 0.05
        # this should be commented out
        # config.model.ssl_feature_dim = 128
        config.model.ssl_feature_dim = 2048
    return config.clone()
        
        
def load_config_from_yaml(config_path):
    config = CfgNode.load_cfg(open(config_path))
    
    return config
    
    
def save_experiment_config(config):
    save_dir = config.system.save_dir
    with open(os.path.join(save_dir, 'experiment_config.yaml'), 'w') as f:
        with redirect_stdout(f):
            print(config.dump())
            
            
def match_config_with_args(config, args):
    config.defrost()
    
    config.dataset.name = args.data
    
    config.system.gpu = args.gpu
    config.system.num_workers = args.num_workers
    config.system.world_size = args.world_size
    config.system.rank = args.rank
    config.system.save_dir = os.path.join(config.system.save_dir, '_'.join([args.experiment_name, args.save_date]))
    config.system.log_dir = os.path.join(config.system.log_dir, '_'.join([args.experiment_name, args.save_date]))
    config.system.dist_url = args.dist_url
    
    config.train.batch_size = args.batch_size
    config.train.num_neg = args.num_neg if args.num_neg is not None else config.train.num_neg
    config.train.base_lr = args.lr
    config.train.tau = args.tau
    config.train.start_epoch = args.start_epoch
    config.train.use_schedule = args.use_schedule
    config.train.schedule = args.schedule
    config.train.temperature = args.temperature
    config.train.wd = args.weight_decay
    config.train.use_eqco = args.use_eqco
    config.train.eqco_k = args.eqco_k
    config.train.use_dcl = args.use_dcl
    config.train.tau_plus = args.tau_plus
    config.train.use_hcl = args.use_hcl
    config.train.beta = args.beta
    config.train.epochs = args.epoch
    config.train.use_symmetric_logit = args.use_symmetric_logit
    config.train.use_temp_schedule = args.use_temp_schedule
    config.train.temp_warmup = args.temp_warmup
    config.train.temp_schedule = args.temp_schedule
    config.train.use_wass = args.use_wass
    config.train.use_neg_wass = args.use_neg_wass
    config.train.moclr_option = args.moclr_option
    config.train.pos_momentum_full = args.pos_momentum_full
    config.train.negcl_option = args.negcl_option
    config.train.num_cluster = args.num_cluster
    # config.train.num_pseudo_class = args.num_pseudo_class
    
    config.model.arch = args.arch
    config.model.bn_encoder = args.bn_encoder
    config.model.regular_pred = args.regular_pred
    config.model.bn_proj = args.bn_proj
    config.model.bn_pred = args.bn_pred
    
    config.method = args.method
    
    if config.method == 'simclr':
        config.train.use_symmetric_logit = args.use_symmetric_logit
        
    if not os.path.exists(config.system.save_dir):
        os.mkdir(config.system.save_dir)
    if not os.path.exists(config.system.log_dir):
        os.mkdir(config.system.log_dir)
    
    if config.dataset.name.startswith('cifar'):
        config.dataset.img_size = [32, 32, 3]

        if config.method == 'simsiam':
            pass
            # config.model.ssl_feature_dim = 512
    
    if config.model.regular_pred:
        config.model.ssl_feature_dim = 128
    # resume
    if args.resume != None:
        epoch, lr, model, arch = get_resume_info(args.resume)
        config.train.start_epoch = epoch
        config.model.arch = arch
        config.train.resume = args.resume
        del epoch
        del lr
        del model
        del arch


if __name__ == "__main__":
    inspect_config(config)
