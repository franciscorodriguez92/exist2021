#%%
import pandas as pd
import os
import ml.utils as utils
import numpy as np
import random
import joblib
import argparse

##% Seed
seed = 123
np.random.seed(seed)
random.seed(seed)

#%%

parser = argparse.ArgumentParser(description = 'EXIST 2021 ML')
parser.add_argument('--test_path', type = str, default = '../data/input/EXIST2021_dataset-test/EXIST2021_dataset/test/EXIST2021_test.tsv', help = 'train_path')
parser.add_argument('--model_path', type = str, default = '../models/ml_test.pkl', help = 'path to save trained model (e.g. ../models/bert_test.pt)')
parser.add_argument('--language', type = str, default = "both", help = 'language (es, en or both)')
parser.add_argument('--text_cleaner', action = 'store_true', default = False, help = 'preprocess text')
parser.add_argument('--task', type = str, default = 'task1', help = 'task (task1 or task2)')
parser.add_argument('--output_path', type = str, default = '../submissions/submission.tsv', help = 'output path for submission file')
parser.add_argument('--sample', action = 'store_true', default = False, help = 'get a sample of 1 percent')


args = parser.parse_args()
#print(args)
#%%
test_path = args.test_path
language = args.language
text_cleaner = args.text_cleaner
sample = args.sample
task = args.task
#classifier = 'svm'
features = ['text']
model_path=args.model_path
output_path = args.output_path

if language == "both":
    model_path_en = model_path.replace('.pkl', '')+'_english.pkl' 


#%%
if language=="both":
    df_test_es = utils.read_file(test_path, "es", sample, text_cleaner)
    df_test_en = utils.read_file(test_path, "en", sample, text_cleaner)
else:
    df_test = utils.read_file(test_path, language, sample, text_cleaner)


#%%
if language == "both":
    pipeline = joblib.load(model_path)
    pipeline_en = joblib.load(model_path_en)
    y_pred_test = pipeline.predict(df_test_es[features])
    y_pred_test_en = pipeline_en.predict(df_test_en[features])
else:
    pipeline = joblib.load(model_path)
    y_pred_test = pipeline.predict(df_test[features])

#%%
if language == "both":
    df_test_es['category']=y_pred_test
    df_test_en['category']=y_pred_test_en
    df = pd.concat([df_test_es, df_test_en])
    df=df[['id', 'test_case', 'category']]
    df.to_csv(output_path.replace('.tsv', '')+'_'+str(task)+'_ml.tsv' , sep="\t", index=False)
else:
    df_test['category']=y_pred_test
    df=df[['id', 'test_case', 'category']]
    df.to_csv(output_path.replace('.tsv', '')+'_'+str(language)+'_'+str(task)+'_ml.tsv' , sep="\t", index=False)
