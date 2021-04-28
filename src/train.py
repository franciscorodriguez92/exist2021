#%% 
import torch
import transformer.datasets as datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import argparse
from transformer.architectures import *
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, AdamW
from transformer.train_transformer import train

# %% 

parser = argparse.ArgumentParser(description = 'EXIST 2021')

parser.add_argument('--basenet', type = str, default = 'bert', help = 'basenet')
parser.add_argument('--train_path', type = str, default = '../data/input/EXIST2021_dataset-test/EXIST2021_dataset/training/EXIST2021_training_split.tsv', help = 'train_path')
parser.add_argument('--val_path', type = str, default = '../data/input/EXIST2021_dataset-test/EXIST2021_dataset/validation/EXIST2021_validation_split.tsv', help = 'val_path')
parser.add_argument('--sample', action = 'store_true', default = False, help = 'get a sample of 1 percent')
parser.add_argument('--batch_size', type = int, default = 16, help = 'training batch-size (default: 16)')
parser.add_argument('--concat_metwo_train', action = 'store_true', default = False, help = 'append metwo dataset to training (task1)')
parser.add_argument('--balance_metwo', action = 'store_true', default = False, help = 'balance metwo when appending to training (task1)')
parser.add_argument('--text_cleaner', action = 'store_true', default = False, help = 'preprocess text')
parser.add_argument('--no_cuda', action = 'store_true',   default = False,        help = 'disables CUDA training')
parser.add_argument('--seed', type = int, default = 123, help = 'random seed (default: 123)')
parser.add_argument('--task', type = str, default = '1', help = 'task (1, 2 or multitask)')
parser.add_argument('--epochs', type = int, default = 20, help = 'number of epochs (default: 20)')
parser.add_argument('--schedule', type = str, default = 'linear', help = 'schedule (linear or constant)')
parser.add_argument('--lr', type = float, default = 2e-5, help = 'learning rate')
parser.add_argument('--model_path', type = str, default = '../models/bert_test.pt', help = 'path to save trained model (e.g. ../models/bert_test.pt)')
parser.add_argument('--learnable_parameter', action = 'store_true', default = False, help = 'learnable parameter when multitasking')

args = parser.parse_args()
#print(args)



#%%
train_path = args.train_path
val_path = args.val_path
sample = args.sample
basenet = args.basenet
batch_size = args.batch_size
concat_metwo_train = args.concat_metwo_train
text_cleaner = args.text_cleaner


args_cuda = args.no_cuda
args_seed = args.seed

if args.task=='1':
    task=1
elif args.task=='2':
    task=2
else:
    task='multitask'

epochs = args.epochs
schedule = args.schedule
learning_rate = args.lr

model_path=args.model_path

balance_metwo = args.balance_metwo
learnable_parameter = args.learnable_parameter
#%% 
train_loader = DataLoader(
    dataset=datasets.exist_2021(train_path, 
    sample = sample, basenet = basenet, 
    concat_metwo=concat_metwo_train, text_cleaner=text_cleaner, balance_metwo=balance_metwo), 
    batch_size=batch_size, shuffle=True)

validation_loader = DataLoader(
    dataset=datasets.exist_2021(val_path, 
    sample = sample, basenet = basenet, text_cleaner=text_cleaner), 
    batch_size=batch_size, shuffle=True)

#%%
use_cuda = not args_cuda and torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args_seed)
if(use_cuda):
	torch.cuda.manual_seed(args_seed)
	torch.backends.cudnn.benchmark = True
np.random.seed(args_seed)

print("\nDevice: " + str(device) +"; Seed: "+str(args_seed))

#%%
if task==1:
    model=create_model(basenet, device)
elif task==2:
    model=create_model(basenet, device, num_labels=6)
elif task=='multitask':
    model=create_model(basenet, device, multitask=True)
elif task=='multitask' and learnable_parameter:
    model=create_model(basenet, device, multitask=True, learnable_parameter=True)
else:
    print("Please enter a valid task")

#%%
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_training_steps = epochs * len(train_loader)
num_warmup_steps = num_training_steps // 10
if schedule == "linear":
    print("Using linear scheduler with warmup")
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
elif schedule == "constant":
    print("Using constant scheduler with warmup")
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps,
    )

#%%

train(model, optimizer, train_loader, validation_loader, epochs, model_path, scheduler=scheduler, task=task)
