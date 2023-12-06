# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import copy
import utils
import argparse
import wandb
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9 import dataset
from qm9.models import get_optim, get_model, get_autoencoder, get_latent_diffusion
from equivariant_diffusion import en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
import torch
import time
import pickle
from qm9.utils import prepare_context, compute_mean_mad
from train_progdistill import train_epoch, test, analyze_and_save

'''
python main_progdistill.py --n_epochs 30 --n_stability_samples 10 --diffusion_noise_schedule polynomial_2 
--diffusion_noise_precision 1e-5 --diffusion_steps 500 --diffusion_loss_type l2 --batch_size 64 --nf 256 
--n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 10 --ema_decay 0.9999 --train_diffusion
--latent_nf 2 --exp_name $student_name --teacher_path outputs/$teacher_model
'''

parser = argparse.ArgumentParser(description='ProgDistillatsion')
parser.add_argument('--exp_name', type=str, default='qm9_500')
args = parser.parse_args()

with open(join(f'outputs/{args.exp_name}', 'args.pickle'), 'rb') as f:
    args = pickle.load(f)

dataset_info = get_dataset_info(args.dataset, args.remove_h)
dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

args.diffusion_steps = 2*args.diffusion_steps

model_state_dict = torch.load(join(args.teacher_path, 'generative_model.npy'))
model_ema_state_dict = torch.load(join(args.teacher_path, 'generative_model_ema.npy'))
optim_state_dict = torch.load(join(args.teacher_path, 'optim.npy'))

model, _, _ = get_latent_diffusion(args, torch.device("cpu"), dataset_info, dataloaders['train'])
model_ema = copy.deepcopy(model)

model.load_state_dict(model_state_dict)
model_ema.load_state_dict(model_ema_state_dict)

args.diffusion_steps = args.diffusion_steps//2

model.gamma = en_diffusion.PredefinedNoiseSchedule(args.diffusion_noise_schedule, 
                                                   args.diffusion_steps, args.diffusion_noise_precision)
model_ema.gamma = en_diffusion.PredefinedNoiseSchedule(args.diffusion_noise_schedule, 
                                                   args.diffusion_steps, args.diffusion_noise_precision)
args.exp_name = args.exp_name+'_gamma_fixed'


utils.create_folders(args)

utils.save_model(model, 'outputs/%s/generative_model.npy' % args.exp_name)
utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
    pickle.dump(args, f)
