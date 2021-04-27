#%%
import pandas as pd
import os
import ml.utils as utils
import numpy as np
import random
import joblib

##% Seed
seed = 123
np.random.seed(seed)
random.seed(seed)

#%%
test_path = '../data/input/EXIST2021_dataset-test/EXIST2021_dataset/test/EXIST2021_test.tsv'
language = "both"
text_cleaner = False
#sample = False
task = "task2"
#classifier = 'svm'
features = ['text']
model_path = '../models/ml_test.pkl'
output_path = '../submissions/submission.tsv'
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
