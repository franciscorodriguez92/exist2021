#%% 
import datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
#%%
# subset_indices = list(range(0,32))
# exist_data = datasets.exist_2021('../../data/input/MeTwo2.tsv', sample = True, basenet = 'roberta')
#%% 
# valid_size = 0.2
# num_train = len(exist_data)
# indices = list(range(num_train))
# split = int(np.floor(valid_size * num_train))
# train_idx, valid_idx = indices[split:], indices[:split]
#%%
train_loader = DataLoader(
    dataset=datasets.exist_2021('../../data/input/EXIST2021_dataset-test/EXIST2021_dataset/training/EXIST2021_training_split.tsv', 
    sample = True, basenet = 'roberta', concat_metwo=False, text_cleaner=False), 
    batch_size=15, shuffle=False)

validation_loader = DataLoader(
    dataset=datasets.exist_2021('../../data/input/EXIST2021_dataset-test/EXIST2021_dataset/validation/EXIST2021_validation_split.tsv', 
    sample = True, basenet = 'roberta'), 
    batch_size=15, shuffle=False)

#%%

# train_loader = DataLoader(
#     dataset=datasets.exist_2021('../../data/input/MeTwo2.tsv', sample = True, basenet = 'roberta'), 
#     sampler=SubsetRandomSampler(train_idx), batch_size=16, shuffle=False)

# validation_loader = DataLoader(
#     dataset=datasets.exist_2021('../../data/input/MeTwo2.tsv', sample = True, basenet = 'roberta'), 
#     sampler=SubsetRandomSampler(valid_idx), batch_size=16, shuffle=False)
#%% 

#dataiter = iter(train_loader)
#data = dataiter.next()
#print(data)
for batch_id, batch_data in enumerate(train_loader):
    print(batch_id)
    if batch_id ==3:
    	print(batch_data[1])
    #print(f'Batch ID: {batch_id}, Batch: {batch_data[0]}, Batch labels task1:
    #  {batch_data[1][0]}, Batch labels task2: {batch_data[1][1]}')
    #print(batch_data[0])
    #break

#%% 
import torch
import numpy as np
args_cuda = True
args_seed = 123
use_cuda = not args_cuda and torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#kwargs   = {'num_workers': 32, 'pin_memory': True} if use_cuda else {}

torch.manual_seed(args_seed)
if(use_cuda):
	torch.cuda.manual_seed(args_seed)
	torch.backends.cudnn.benchmark = True
np.random.seed(args_seed)

print("\nDevice: " + str(device) +"; Seed: "+str(args_seed))

#%%
from architectures import *
model = create_model('roberta', device)
#model = create_model('roberta', device, multitask=True)
#model = create_model('roberta', device, num_labels=6)

#%%
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, AdamW
epochs = 1
schedule = 'linear'
optimizer = AdamW(model.parameters(), lr=2e-5)
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
# from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, AdamW
# epochs = 1
# accumulation_steps=2
# batch_size = 16
# param_optimizer = list(model.named_parameters())
# no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
# optimizer_grouped_parameters = [
#   {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#   {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
#   ]
# num_train_optimization_steps = int(epochs*len(train_loader.dataset.data)/batch_size/accumulation_steps)
# #num_train_optimization_steps = int(len(train_data_loader)) * epochs
# optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
# scheduler = get_linear_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=.1,
#     num_training_steps=num_train_optimization_steps
# )


#%% 
#torch.save(model, '../../models/bert_test.pt')
from train_transformer import *
model_path='../../models/bert_test.pt'
train(model, optimizer, train_loader, validation_loader, epochs, model_path, scheduler=scheduler, task=1)


#%%

model2=load_model('../../models/bert_test.pt', device)
(model.state_dict()['classifier.out_proj.weight'].numpy()) == (model2.state_dict()['classifier.out_proj.weight'].numpy())



#%%
from utils import *
generate_submission(
    model_path='../../models/bert_test.pt', basenet='roberta', device=device, test_path='../../data/input/MeTwo2.tsv',
    output_path='../../submissions/submission.tsv')


# %%
#df = pd.read_table('../../data/input/MeTwo2.tsv', sep="\t")
#df = pd.read_table('../../data/input/EXIST2021_dataset-test/EXIST2021_dataset/test/EXIST2021_test.tsv', sep="\t")
#%%

#%%
import torch
from torch.nn import CrossEntropyLoss
from transformers import BertModel, RobertaModel
#https://github.com/huggingface/transformers/blob/424419f54964a5ca68277e700a8264f701968639/src/transformers/models/bert/modeling_bert.py
class BERT_based_classifier_multitask(torch.nn.Module):
	def __init__(self, basenet= 'bert', bert_freeze= False, num_labels= 2, num_labels_task2=6):
		super(BERT_based_classifier_multitask, self).__init__()
		self.num_labels = num_labels
		self.num_labels_task2 = num_labels_task2
		self.basenet = basenet
		if basenet == 'bert':
			print("Base architecture: bert-base-multilingual-cased")
			self.encoder = BertModel.from_pretrained('bert-base-multilingual-cased',
													  output_hidden_states = False 
													)
		elif basenet == 'roberta':
			print("Base architecture: xlm-roberta-base")
			self.encoder = RobertaModel.from_pretrained('xlm-roberta-base',
													  output_hidden_states = False 
													)
		if bert_freeze:
			print("Freezing BERT!")
			# freeze bert so that is is not finetuned
			for name, param in self.encoder.named_parameters():                
				if param.requires_grad is not None:
					param.requires_grad = False
		self.dropout = torch.nn.Dropout(0.1)
		self.classifier1 = torch.nn.Linear(in_features= 768, out_features= num_labels)
		self.classifier2 = torch.nn.Linear(in_features= 768, out_features= num_labels_task2)
	
	def forward(self, input_id, mask_id, token_type_id, labels=None):
		pooled_output = self.encoder(input_ids= input_id, attention_mask= mask_id, token_type_ids= token_type_id)['pooler_output']
		pooled_output = self.dropout(pooled_output)
		logits_task1 = self.classifier1(pooled_output)
		logits_task2 = self.classifier2(pooled_output)
		# print('LOGITS:::::::: ', logits_task1)
		# print('LOGITS shape:::::::: ', logits_task1.shape)
		# print('LOGITS:::::::: ', logits_task2)
		# print('LOGITS shape:::::::: ', logits_task2.shape)
		if labels is not None:
			loss_fct = CrossEntropyLoss(ignore_index=-1)
			loss1 = loss_fct(logits_task1.view(-1, self.num_labels), labels[:,0].view(-1))
			loss2 = loss_fct(logits_task2.view(-1, self.num_labels_task2), labels[:,1].view(-1))
			loss = loss1 + loss2
			#print("LOSS::::::::::::::", loss.item())
			return loss
		else:
			return [logits_task1, logits_task2]


model = BERT_based_classifier_multitask(basenet='roberta')


#%%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for step, batch in enumerate(train_loader):
    #b_input_ids, b_input_mask, b_labels = batch
    data = batch[0]
    b_input_ids = data[0].squeeze()
    b_input_mask = data[1].squeeze()
    #b_labels = batch[1][:,0].squeeze()
    b_labels = batch[1].squeeze()
    #print(b_labels)
    #print(b_input_ids)
    b_input_ids = b_input_ids.to(device, dtype=torch.long)
    b_input_mask = b_input_mask.to(device, dtype=torch.long)
    b_labels = b_labels.to(device, dtype=torch.long)
    #loss = model(input_ids=b_input_ids, token_type_ids=None, attention_mask=b_input_mask,labels=b_labels)[0]
    #loss = model(input_ids=b_input_ids, token_type_ids=None, attention_mask=b_input_mask,labels=b_labels)[0]
    #Multitask::::::!!!!!!!
    loss = model(input_id=b_input_ids, token_type_id=None, mask_id=b_input_mask,labels=b_labels)
    logits_task1, logits_task2 = model(input_id=b_input_ids, token_type_id=None, mask_id=b_input_mask)
    loss.backward()
    optimizer.step()
    #break

print('finished')
#%% 
x = torch.randn(10, 5)
target = torch.randint(0, 5, (10,))

criterion_weighted = nn.CrossEntropyLoss(ignore_index=4)
loss_weighted = criterion_weighted(x, target)

criterion_weighted_manual = nn.CrossEntropyLoss(reduction='none', ignore_index=4)
loss_weighted_manual = criterion_weighted_manual(x, target)
print(target)
print(loss_weighted_manual)
print(loss_weighted_manual.sum())
loss_weighted_manual = loss_weighted_manual.mean()
print(loss_weighted_manual)
print(loss_weighted)
print(loss_weighted == loss_weighted_manual)
# %%
