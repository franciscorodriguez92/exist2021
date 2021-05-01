import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, BertTokenizer, RobertaTokenizer, DebertaTokenizer

import sys

import re
import unidecode
import string
from sklearn.utils import resample


def downsample(df, clase, n):
    df_majority = df[df.categoria==clase]
    df_minority = df[df.categoria!=clase]
    df_majority_downsampled = resample(df_majority, 
                                 replace=False,
                                 n_samples=n,
                                 random_state=123)
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    df_downsampled = df_downsampled.sample(frac=1, random_state = 123).reset_index(drop=True)
    return(df_downsampled)

class TextCleaner(object):

    def __init__(self, filter_users=False, filter_hashtags=False,
                 filter_urls=False, convert_hastags=False,
                 lowercase=True, replace_exclamation=False,
                 replace_interrogation=False,
                 remove_accents=False,
                 remove_punctuation=False):  
        self.filter_users = filter_users
        self.filter_hashtags = filter_hashtags
        self.filter_urls = filter_urls
        self.convert_hastags = convert_hastags
        self.lowercase = lowercase
        self.replace_exclamation = replace_exclamation
        self.replace_interrogation = replace_interrogation
        self.remove_accents = remove_accents
        self.remove_punctuation = remove_punctuation

    def __call__(self, text):
        #text = text.decode('utf-8') 
        if self.filter_urls:
            text = self.replace_url(text)  
        if self.remove_accents:
            text = self.strip_accents(text)        
        if self.filter_users:
            text = self.replace_user(text)
        if self.convert_hastags:
            text = self.convert_hastags_upper(text)             
        if self.filter_hashtags:
            text = self.replace_hastags(text)          
        if self.replace_exclamation:
            text = self.replace_exclamations(text) 
        if self.replace_interrogation:
            text = self.replace_interrogations(text)
        if self.remove_punctuation:
            text = self.filter_punctuation(text)  
        if self.lowercase:
            text = self.to_lower_endline(text)    
        return text
    

    def to_lower_endline(self, text):
        return text.lower().replace('\n',' ')
    
    def filter_punctuation(self, text):
        #return text.translate(None, string.punctuation)
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def strip_accents(self, text):
        return unidecode.unidecode(text)

    def replace_user(self, text):
        return re.sub( r"(?<=^|(?<=[^a-zA-Z0-9-\\.]))@([A-Za-z]+[A-Za-z0-9]+)", r"twuser", text)
    
    def replace_hastags(self, text):
        return re.sub( r"(?<=^|(?<=[^a-zA-Z0-9-\\.]))#([A-Za-z]+[A-Za-z0-9]+)", r"twhastag", text)
    
    def convert_hastags_upper(self, text):
        return re.sub( r"([A-Z])", r" \1", text)
    
    def replace_exclamations(self, text):
        return re.sub( r"(!+|¡+)", r" twexclamation", text)
    
    def replace_url(self, text):
        return re.sub( r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", r"twurl", text)
    
    def replace_interrogations(self, text):
        text = text.replace(u'¿', 'twinterrogation ')
        return re.sub( r"(\?+)", r" twinterrogation", text)


class exist_2021(torch.utils.data.Dataset):
	'''
	Hahackathon dataset

	filename: train/val/test file to be read

	basenet : bert/ernie/roberta/deberta

	is_test : if the input file does not have groundtruth labels
	        : (for evaluation on the leaderboard)
	'''

	def __init__(self, filename, basenet= 'bert', max_length= 128, stop_words= False, is_test= False, language=False, sample=False, concat_metwo=False, text_cleaner=False, balance_metwo=False):
		super(exist_2021, self).__init__()

		self.is_test = is_test

		if stop_words:
			#self.nlp  = English()
			#TODO: preprocesado
			pass

		self.data    = self.read_file(filename, stop_words, language, sample, concat_metwo, text_cleaner, balance_metwo)

		if basenet == 'bert':
			print("Tokenizer: bert-base-multilingual-cased\n")
			self.token = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
		if basenet == 'roberta':
			print("Tokenizer: xlm-roberta-base\n")
			self.token = AutoTokenizer.from_pretrained('xlm-roberta-base')
		if basenet == 'roberta_twitter':
			print("Tokenizer: twitter-xlm-roberta-base\n")
			self.token = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base")
		# elif basenet == 'ernie':
		# 	print("Tokenizer: ernie-2.0-en\n")
		# 	self.token = AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-en")
		# elif basenet == 'roberta':
		# 	print("Tokenizer: roberta-base\n")
		# 	self.token = RobertaTokenizer.from_pretrained('roberta-base')
		# elif basenet == 'deberta':
		# 	print("Tokenizer: microsoft/deberta-base\n")
		# 	self.token = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
		
		self.max_length = max_length
		#self.segment_id = torch.tensor([1] * self.max_length).view(1, -1)

		
	def read_file(self, filename, stop_words, language, sample, concat_metwo, text_cleaner, balance_metwo):
		#df = pd.read_table(filename, sep="\t", dtype={'id': 'str'})
		df = pd.read_table(filename, sep="\t")
		if language == "es":
			df = df[df['language'] == "es"]
		elif language == "en":
			df = df[df['language'] == "en"]
		
		if not self.is_test:
			df['task1']=df['task1'].map({'non-sexist' : 0, 'sexist': 1})
			df['task2']=df['task2'].map({'non-sexist' : 0, 'ideological-inequality': 1, 'stereotyping-dominance': 2, 'objectification': 3, 'sexual-violence': 4, 'misogyny-non-sexual-violence': 5})
			
		if concat_metwo:
			path = '../data/input/metwo/'
			labels = pd.read_table(path + 'corpus_machismo_etiquetas.csv', sep=";", dtype={'status_id': 'str'})
			labels = labels[["status_id","categoria"]]
			tweets_fields = pd.read_csv(path + 'corpus_machismo_frodriguez_atributos_extra.csv', 
										dtype={'status_id': 'str'})
			tweets_fields = tweets_fields[['status_id','text']]
			metwo = tweets_fields.merge(labels, on = 'status_id', how = 'inner')
			metwo = metwo[metwo['categoria']!='DUDOSO']
			if balance_metwo:
				a = dict(metwo['categoria'].value_counts())
				metwo = downsample(metwo, 'NO_MACHISTA', a['MACHISTA'])
			# test_case	id	source	language	text	task1	task2
			#df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'}, inplace=True)
			metwo['test_case']='EXIST2021'
			metwo['id']=metwo['status_id']
			metwo['source']='twitter'
			metwo['language']='es'
			metwo['task1']=metwo['categoria'].map({'NO_MACHISTA' : 0, 'MACHISTA': 1})
			metwo['task2']=-1
			metwo=metwo[['test_case', 'id', 'source', 'language', 'text', 'task1', 'task2']]
			df = df.append(metwo, ignore_index = True)

		if sample:
			df=df.sample(frac=0.01, random_state=123)
		
		if text_cleaner:
			preprocessor = TextCleaner(filter_users=True, filter_hashtags=True, 
                           filter_urls=True, lowercase=True, 
                           replace_exclamation=True, replace_interrogation=True, 
                           remove_punctuation=True)
			df['text'] = df['text'].apply(lambda row: preprocessor(row))
			

		# # removing stop-words
		# # https://www.analyticsvidhya.com/blog/2019/08/how-to-remove-stopwords-text-normalization-nltk-spacy-gensim-python/
		# if stop_words:
		# 	for i in range(len(df)):
		# 		text = df.iloc[i]['text']
		# 		text = text.split(' ')
		# 		filtered_text =[] 

		# 		for word in text:
		# 			lexeme = self.nlp.vocab[word]
		# 			if lexeme.is_stop == False:
		# 				filtered_text.append(word) 
		# 		# print("original", text)
		# 		text = ' '.join(filtered_text)
		# 		# print("after", filtered_text)
		# 		# print("after", text)
		# 		df.loc[i, 'text'] = text

		# # replace all NaN with 0
		# # will be used dring loss function computation
		# df = df.fillna(0)

		# if not self.is_test:
		# 	df['humor_controversy'] = df['humor_controversy'].astype('int')

		print(df.shape)
		print("Sampled input from the file: {}".format(filename))
		print(df.head())

		return df

	
	def get_tokenized_text(self, text):		
		# marked_text = "[CLS] " + text + " [SEP]"
		encoded = self.token(text= text,  					# the sentence to be encoded
							 add_special_tokens= True,  	# add [CLS] and [SEP]
							 max_length= self.max_length,  	# maximum length of a sentence
							 padding= 'max_length',  		# add [PAD]s
							 return_attention_mask= True,  	# generate the attention mask
							 return_tensors = 'pt',  		# return PyTorch tensors
							 truncation= True
							) 

		input_id = encoded['input_ids']
		mask_id  = encoded['attention_mask']

		return input_id, mask_id

		
	def __len__(self):
		return len(self.data)
	

	def __getitem__(self, idx):
		
		text  = self.data.iloc[idx]['text']

		label = []

		if not self.is_test:
			label.append(self.data.iloc[idx]['task1'])
			label.append(self.data.iloc[idx]['task2'])


		else:
			label.append(self.data.iloc[idx]['id'])

		label = torch.tensor(label)

		input_id, mask_id  = self.get_tokenized_text(text)
		
		return [input_id, mask_id], label