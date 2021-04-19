#%%
from transformer.utils import generate_submission
import torch
import numpy as np
import argparse

#%%
parser = argparse.ArgumentParser(description = 'Generate submissions EXIST 2021')

parser.add_argument('--basenet', type = str, default = 'bert', help = 'basenet')
parser.add_argument('--model_path', type = str, default = '../models/bert_test.pt', help = 'path to save trained model (e.g. ../models/bert_test.pt)')
parser.add_argument('--test_path', type = str, default = '../data/input/EXIST2021_dataset-test/EXIST2021_dataset/test/EXIST2021_test.tsv', help = 'train_path')
parser.add_argument('--output_path', type = str, default = '../submissions/submission.tsv', help = 'output path for submission file')
parser.add_argument('--task', type = str, default = '1', help = 'task (1, 2 or multitask)')
parser.add_argument('--batch_size', type = int, default = 2, help = 'batch-size (default: 2)')
parser.add_argument('--sample', action = 'store_true', default = True, help = 'get a sample of 1 percent')
parser.add_argument('--no_cuda', action = 'store_true',   default = False,        help = 'disables CUDA training')
parser.add_argument('--seed', type = int, default = 123, help = 'random seed (default: 123)')

args = parser.parse_args()


#%%
model_path = args.model_path
basenet = args.basenet
test_path = args.test_path
output_path = args.output_path
batch_size = args.batch_size
sample = args.sample
args_cuda = args.no_cuda
args_seed = args.seed

if args.task=='1':
    task=1
elif args.task=='2':
    task=2
else:
    task='multitask'
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
generate_submission(
    model_path, basenet, device, test_path, output_path, task, batch_size, sample)    
