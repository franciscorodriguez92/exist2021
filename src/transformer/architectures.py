import transformers
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, AutoModelForSequenceClassification
import torch
from torch.nn import CrossEntropyLoss
from transformers import BertModel, RobertaModel, AutoModel

def create_model(model_name, device, num_labels=2, multitask=False, learnable_parameter=False, focal_loss=False):
    # if model_name not in AVAILABLE_MODELS:
    #     raise ValueError(f"{model_name} not available -- must be in {AVAILABLE_MODELS}")

    if model_name == "bert_uncased":
        bert_name = "bert-base-multilingual-uncased"
    elif model_name == "bert":
        bert_name = "bert-base-multilingual-cased"
    elif model_name == "roberta":
        bert_name = "xlm-roberta-base"
    elif model_name == "roberta_twitter":
        bert_name = "cardiffnlp/twitter-xlm-roberta-base"

    print(f"Using {bert_name}")
    if model_name == "roberta" and not multitask:
        bert_model = RobertaForSequenceClassification.from_pretrained(bert_name, num_labels=num_labels)
    elif model_name == "bert" and not multitask:
        bert_model = BertForSequenceClassification.from_pretrained(bert_name, num_labels=num_labels)
    elif model_name == "roberta_twitter" and not multitask:
        bert_model = AutoModelForSequenceClassification.from_pretrained(bert_name, num_labels=num_labels)   
	#bert_tokenizer = BertTokenizer.from_pretrained(bert_name)
    elif multitask and not learnable_parameter:
        bert_model = BERT_based_classifier_multitask(basenet=model_name, focal_loss=focal_loss)
    elif multitask and learnable_parameter:
        bert_model = BERT_based_classifier_multitask_alpha_parameter(basenet=model_name, focal_loss=focal_loss)

    bert_model = bert_model.to(device)

    return bert_model


class FocalLoss(torch.nn.Module):
    """Multi-class Focal loss implementation"""
    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = torch.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = torch.nn.functional.nll_loss(log_pt, target, self.weight, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss

class BERT_based_classifier_multitask_alpha_parameter(torch.nn.Module):
	def __init__(self, basenet= 'bert', bert_freeze= False, num_labels= 2, num_labels_task2=6, focal_loss = False):
		super(BERT_based_classifier_multitask_alpha_parameter, self).__init__()
		self.num_labels = num_labels
		self.num_labels_task2 = num_labels_task2
		self.basenet = basenet
		self.alpha = torch.nn.Parameter(torch.tensor([0.5]))
		self.focal_loss = focal_loss
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
		elif basenet == 'roberta_twitter':
			print("Base architecture: twitter-xlm-roberta-base")
			self.encoder = AutoModel.from_pretrained('cardiffnlp/twitter-xlm-roberta-base',
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
			if self.focal_loss:
				loss_fct = FocalLoss(ignore_index=-1)
			else:
				loss_fct = CrossEntropyLoss(ignore_index=-1)			
			loss1 = loss_fct(logits_task1.view(-1, self.num_labels), labels[:,0].view(-1))
			loss2 = loss_fct(logits_task2.view(-1, self.num_labels_task2), labels[:,1].view(-1))
			loss = self.alpha*loss1 + (1-self.alpha)*loss2
			#print(self.alpha.item())
			#print("LOSS::::::::::::::", loss.item())
			return loss
		else:
			return [logits_task1, logits_task2]

class BERT_based_classifier_multitask(torch.nn.Module):
	def __init__(self, basenet= 'bert', bert_freeze= False, num_labels= 2, num_labels_task2=6, focal_loss = False):
		super(BERT_based_classifier_multitask, self).__init__()
		self.num_labels = num_labels
		self.num_labels_task2 = num_labels_task2
		self.basenet = basenet
		self.focal_loss = focal_loss
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
		elif basenet == 'roberta_twitter':
			print("Base architecture: twitter-xlm-roberta-base")
			self.encoder = AutoModel.from_pretrained('cardiffnlp/twitter-xlm-roberta-base',
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
			if self.focal_loss:
				loss_fct = FocalLoss(ignore_index=-1)
			else:
				loss_fct = CrossEntropyLoss(ignore_index=-1)
			loss1 = loss_fct(logits_task1.view(-1, self.num_labels), labels[:,0].view(-1))
			loss2 = loss_fct(logits_task2.view(-1, self.num_labels_task2), labels[:,1].view(-1))
			loss = loss1 + loss2
			#print("LOSS::::::::::::::", loss.item())
			return loss
		else:
			return [logits_task1, logits_task2]