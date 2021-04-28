# -*- coding: utf-8 -*-
#%% imports
import pandas as pd
import os
import ml.utils as utils
import ml.classifier as clf
import numpy as np
import random
import joblib

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, cross_val_predict
from ml.preprocess import TextCleaner
from ml.preprocess import ColumnSelector
from ml.preprocess import TypeSelector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report

import argparse

##% Seed
seed = 123
np.random.seed(seed)
random.seed(seed)

#%%

parser = argparse.ArgumentParser(description = 'EXIST 2021 ML')

parser.add_argument('--train_path', type = str, default = '../data/input/EXIST2021_dataset-test/EXIST2021_dataset/training/EXIST2021_training_split.tsv', help = 'train_path')
parser.add_argument('--val_path', type = str, default = '../data/input/EXIST2021_dataset-test/EXIST2021_dataset/validation/EXIST2021_validation_split.tsv', help = 'val_path')
parser.add_argument('--task', type = str, default = 'task1', help = 'task (task1 or task2)')
parser.add_argument('--language', type = str, default = "both", help = 'language (es, en or both)')
parser.add_argument('--text_cleaner', action = 'store_true', default = False, help = 'preprocess text')
parser.add_argument('--sample', action = 'store_true', default = False, help = 'get a sample of 1 percent')
parser.add_argument('--classifier', type = str, default = 'svm', help = 'svm, random_forest or logistic_regression')
parser.add_argument('--model_path', type = str, default = '../models/ml_test.pkl', help = 'path to save trained model (e.g. ../models/bert_test.pt)')

args = parser.parse_args()
#print(args)
#%% Inputs
train_path = args.train_path
val_path = args.val_path
sample = args.sample
language = args.language
text_cleaner = args.text_cleaner
task = args.task
classifier = args.classifier
features = ['text']
model_path=args.model_path
if language == "both":
    model_path_english = model_path.replace('.pkl', '')+'_english.pkl' 

#use_lexicon = 'polar'
# use_lexicon = 'metwo'
# #use_lexicon = 'hurtlex'
# #use_lexicon = 'hb'
# tokens_no_lexicon = False

# add_vector_lexicon = True
# pca_components = 0

# pca_to_tfidf = False
# pca_components_to_tfidf = 10

# exclude_tfidf_features = False

# exclude_extra_features = False



#%%

if language=="both":
    df_train_es = utils.read_file(train_path, "es", sample, text_cleaner)
    df_val_es = utils.read_file(val_path, "es", sample, text_cleaner)
    df_train_en = utils.read_file(train_path, "en", sample, text_cleaner)
    df_val_en = utils.read_file(val_path, "en", sample, text_cleaner)
else:
    df_train = utils.read_file(train_path, language, sample, text_cleaner)
    df_val = utils.read_file(val_path, language, sample, text_cleaner)

#%% Read files
#path = os.getcwd()

# metwo2 = pd.read_table('../data/input/MeTwo2.tsv', sep="\t", 
#                        dtype={'id': 'str'})
# if language == "es":
#     tweets_labeled = metwo2[metwo2['language'] == "es"]
# elif language == "en":
#     tweets_labeled = metwo2[metwo2['language'] == "en"]


#%%
# labels = labels[["status_id","categoria"]]
# tweets_fields = pd.read_csv(path + '/resources/data/corpus_machismo_frodriguez_atributos_extra.csv', 
#                             dtype={'status_id': 'str'})

# #%% Cruce de los ficheros

# tweets_fields = utils.change_dtypes(tweets_fields, {'status_id': str})
# labels = utils.change_dtypes(labels, {'status_id': str})
# tweets_labeled = tweets_fields.merge(labels, on = 'status_id', how = 'inner')
# tweets_labeled['respuesta'] = np.where(tweets_labeled['reply_to_status_id'].isnull(), 'no', 'si')
# tweets_labeled['respuesta_screen_name'] = np.where(tweets_labeled['reply_to_screen_name'].isnull(), 'no', 'si') 
# tweets_labeled['hastag_presence'] = np.where(tweets_labeled['hashtags'].isnull(), 'no', 'si') 
# tweets_labeled['url_presence'] = np.where(tweets_labeled['urls_url'].isnull(), 'no', 'si') 
# tweets_labeled['mentions_presence'] = np.where(tweets_labeled['mentions_user_id'].isnull(), 'no', 'si') 

# dudosos_dict = {"DUDOSO": "MACHISTA"}
# #tweets_labeled = tweets_labeled.replace({"categoria" :dudosos_dict})
# #tweets_labeled = utils.downsample(tweets_labeled, 'NO_MACHISTA', 267)
# #tweets_labeled = utils.downsample(tweets_labeled, 'MACHISTA', 267)
# #tweets_labeled = tweets_labeled.loc[80:100,:]
# #tweets_labeled = tweets_labeled.loc[0:100,:]


# #%% lexicon hurtlex
# #tweets_labeled['hurtlex_lexicon'] = tweets_labeled['text'].apply(lambda row: utils.is_in_hatebase(row))
# if add_vector_lexicon:
#     lexicon_features, tweets_labeled = utils.add_vector_lexicon(use_lexicon, tweets_labeled, pca_components)
# elif use_lexicon == 'hurtlex' and tokens_no_lexicon:
#     tweets_labeled[[use_lexicon,'tokens_no_lexicon']] = tweets_labeled['text'].apply(lambda row: pd.Series([utils.n_in_hurtlex_lexicon(row, tokens_no_lexicon)[0],utils.n_in_hurtlex_lexicon(row, tokens_no_lexicon)[1]]))
# elif use_lexicon == 'hurtlex':
#     tweets_labeled[use_lexicon] = tweets_labeled['text'].apply(lambda row: utils.n_in_hurtlex_lexicon(row))
# elif use_lexicon == 'hb' and tokens_no_lexicon:
#     tweets_labeled[[use_lexicon,'tokens_no_lexicon']] = tweets_labeled['text'].apply(lambda row: pd.Series([utils.n_in_hatebase(row, tokens_no_lexicon)[0],utils.n_in_hatebase(row, tokens_no_lexicon)[1]]))
# elif use_lexicon == 'metwo' and tokens_no_lexicon:
#     tweets_labeled[[use_lexicon,'tokens_no_lexicon']] = tweets_labeled['text'].apply(lambda row: pd.Series([utils.n_in_metwo_lexicon(row, tokens_no_lexicon)[0],utils.n_in_metwo_lexicon(row, tokens_no_lexicon)[1]]))
# elif use_lexicon == 'polar' and tokens_no_lexicon:
#     tweets_labeled[[use_lexicon,'tokens_no_lexicon']] = tweets_labeled['text'].apply(lambda row: pd.Series([utils.n_in_polar_lexicon(row, tokens_no_lexicon)[0],utils.n_in_polar_lexicon(row, tokens_no_lexicon)[1]]))
# elif use_lexicon == 'metwo':
#     tweets_labeled[use_lexicon] = tweets_labeled['text'].apply(lambda row: utils.is_in_metwo_lexicon(row))
# elif use_lexicon == 'polar':
#     tweets_labeled[use_lexicon] = tweets_labeled['text'].apply(lambda row: utils.is_in_polar_lexicon(row))
# else:
#     tweets_labeled[use_lexicon] = tweets_labeled['text'].apply(lambda row: utils.is_in_hatebase(row))

# #%%

# if exclude_extra_features:
#     categorical_features = []
#     x_cols2= ['text']
#     x_cols = []
# else:
#     categorical_features = ['source', 'respuesta', 'respuesta_screen_name',
#                   'hastag_presence', 'url_presence',
#                   'media_type', 'mentions_presence', 'verified']
        
#     x_cols2 = ['text','source', 'display_text_width', 'respuesta', 'respuesta_screen_name',
#                   'favorite_count', 'retweet_count', 'hastag_presence',
#                   'url_presence', 'media_type', 'mentions_presence',
#                   'followers_count', 'friends_count', 'listed_count', 'statuses_count',
#                   'favourites_count', 'verified']
        
#     x_cols = ['source', 'display_text_width', 'respuesta', 'respuesta_screen_name',
#                   'favorite_count', 'retweet_count', 'hastag_presence',
#                   'url_presence', 'media_type', 'mentions_presence',
#                   'followers_count', 'friends_count', 'listed_count', 'statuses_count',
#                   'favourites_count', 'verified']    

# if use_lexicon and add_vector_lexicon is False and tokens_no_lexicon is False:
#     categorical_features.append(use_lexicon)
#     x_cols2.append(use_lexicon)
#     x_cols.append(use_lexicon)

# elif use_lexicon and tokens_no_lexicon:
#     categorical_features.append(use_lexicon)
#     x_cols2.append(use_lexicon)
#     x_cols2.append('tokens_no_lexicon')
#     x_cols.append(use_lexicon)
#     x_cols.append('tokens_no_lexicon')

# else:
#     x_cols2 += lexicon_features
#     x_cols += lexicon_features

# #%% 
# for f in categorical_features:
#     tweets_labeled[f] = tweets_labeled[f].astype("category")

#%% 
# if exclude_extra_features:
#     preprocess_pipeline = make_pipeline(
#         ColumnSelector(columns=x_cols),
#         FeatureUnion(transformer_list=[
#             ("numeric_features", make_pipeline(
#                 TypeSelector(np.number),
#                 SimpleImputer(strategy="constant"),
#                 StandardScaler()
#             ))
#         ])
#     )    
# else:
#     preprocess_pipeline = make_pipeline(
#         ColumnSelector(columns=x_cols),
#         FeatureUnion(transformer_list=[
#             ("numeric_features", make_pipeline(
#                 TypeSelector(np.number),
#                 SimpleImputer(strategy="constant"),
#                 StandardScaler()
#             )),
#             ("categorical_features", make_pipeline(
#                 TypeSelector("category"),
#                 SimpleImputer(strategy="constant", fill_value = "NA"),
#                 OneHotEncoder(handle_unknown='ignore')
#             ))
#         ])
#     )
        


# if pca_to_tfidf:
#     print('PCA to TFIDF')
#     text_pipeline = Pipeline([
#         ('column_selection', ColumnSelector('text')),
#         ('tfidf', TfidfVectorizer(tokenizer=utils.tokenizer_, 
#                                               smooth_idf=True, preprocessor = preprocessor,
#                                               norm=None, min_df=0.01, ngram_range=(1,1))),
#         ('pca', TruncatedSVD(n_components=pca_components_to_tfidf, n_iter=7, random_state=42))
#     ])
    
# else:
#     text_pipeline = Pipeline([
#         ('column_selection', ColumnSelector('text')),
#         ('tfidf', TfidfVectorizer(tokenizer=utils.tokenizer_, 
#                                               smooth_idf=True, preprocessor = preprocessor,
#                                               norm=None, min_df=0.01, ngram_range=(1,1)))
#     ])  

# if exclude_tfidf_features:
#     print('No TFIDF features')
#     classifier_pipeline = Pipeline([('other-features', preprocess_pipeline),
#                           ('clf', clf.get_classifier(classifier))
#                           ])
# else:
#     classifier_pipeline = Pipeline([('feature-union', FeatureUnion([('text-features', text_pipeline), 
#                                    ('other-features', preprocess_pipeline)
#                                   ])),
#                               ('clf', clf.get_classifier(classifier))
#                               ])    


#classifier_pipeline.fit(tweets_labeled[x_cols2], tweets_labeled['categoria'])

#%% Pipeline para baseline    
# preprocessor = TextCleaner(filter_users=True, filter_hashtags=True, 
#                            filter_urls=True, convert_hastags=True, lowercase=True, 
#                            replace_exclamation=True, replace_interrogation=True, 
#                            remove_accents=True, remove_punctuation=True, replace_emojis=False)

text_pipeline = Pipeline([
    ('column_selection', ColumnSelector('text')),
    ('tfidf', TfidfVectorizer(tokenizer=utils.tokenizer_, 
                                          smooth_idf=True, #preprocessor = preprocessor,
                                          norm=None, min_df=0.01, ngram_range=(1,1)))
])

text_pipeline2 = Pipeline([
    ('column_selection', ColumnSelector('text')),
    ('tfidf', TfidfVectorizer(tokenizer=utils.tokenizer_, 
                                          smooth_idf=True, #preprocessor = preprocessor,
                                          norm=None, min_df=0.01, ngram_range=(1,1)))
])
# text_pipeline = Pipeline([
    # ('column_selection', ColumnSelector('text')),
    # ('bow', CountVectorizer(tokenizer=utils.tokenizer_, #preprocessor = preprocessor,
                                          # min_df=0.01, ngram_range=(1,1)))
# ])
baseline_pipeline = Pipeline([('text_pipeline', text_pipeline),
                          ('clf', clf.get_classifier(classifier))
                          ])

baseline_pipeline_en = Pipeline([('text_pipeline', text_pipeline2),
                          ('clf', clf.get_classifier(classifier))
                          ])
#%% Cross validation baseline
# =============================================================================
# print(cross_val_score(baseline_pipeline, tweets_labeled[x_cols2], tweets_labeled['categoria'], cv = 10, n_jobs = 1))
# 
# =============================================================================
#%% Cross validation
# =============================================================================
# print(cross_val_score(classifier_pipeline, tweets_labeled[x_cols2], tweets_labeled['categoria'], cv = 10, n_jobs = 1))
# 
# =============================================================================

#%% Matriz de confusión
#predicted = classifier_pipeline.predict(texto_prueba.drop('categoria', axis=1))
#print np.mean(predicted == texto_prueba['categoria']) 

# =============================================================================
# y_pred = cross_val_predict(classifier_pipeline, tweets_labeled[x_cols2], tweets_labeled['categoria'], cv=10)
# unique_label = np.unique(tweets_labeled['categoria'])
# print(pd.DataFrame(confusion_matrix(tweets_labeled['categoria'], y_pred, labels=unique_label), 
#                    index=['true:{:}'.format(x) for x in unique_label], 
#                    columns=['pred:{:}'.format(x) for x in unique_label]))
# =============================================================================

#%% GridSearchCV

# para chequear los parámetros:: classifier_pipeline.get_params().keys()

# =============================================================================
# import time
# start = time.time()
# parameters = utils.get_grid_parameters(classifier)
# 
# model = GridSearchCV(classifier_pipeline, param_grid=parameters, cv=5,
#                          scoring='accuracy', verbose=1, n_jobs = -1)
# 
# model.fit(tweets_labeled[x_cols2], tweets_labeled['categoria'])
# print("Best score: %0.3f" % model.best_score_)
# print("Best parameters set:")
# best_parameters = model.best_estimator_.get_params()
# for param_name in sorted(parameters.keys()):
#     print("\t%s: %r" % (param_name, best_parameters[param_name]))
# 
# end = time.time()
# print(end - start)
# 
# =============================================================================


#%% Método de evaluación 2: Cross Validation con parámetros por defecto
import time
start = time.time()
scoring = {'acc': 'accuracy',
           'precision': 'precision_macro',
           'recall': 'recall_macro',
           'f1': 'f1_macro'
            }

# test_score_cv = cross_validate(baseline_pipeline, tweets_labeled[features], 
                               # tweets_labeled[task], cv = 10, 
                               # n_jobs = -1, scoring=scoring)

# print('Resultados cross_validate::::::')
# print('ACCURACY::: ', np.mean(test_score_cv['test_acc']))
# print('F1::: ', np.mean(test_score_cv['test_f1']))
# print('RECALL::: ', np.mean(test_score_cv['test_recall']))
# print('PRECISION::: ', np.mean(test_score_cv['test_precision']))

# y_pred = cross_val_predict(baseline_pipeline, tweets_labeled[features], 
#                                tweets_labeled[task], cv=10, n_jobs = -1)
if language != "both":
    baseline_pipeline.fit(df_train[features], df_train[task])
    y_pred = baseline_pipeline.predict(df_val[features])

    unique_label = np.unique(df_train[task])

    print("Classification report:::")
    print(classification_report(df_val[task], y_pred, target_names=unique_label))
    print(f'Saving model at {model_path}')
    joblib.dump(baseline_pipeline, model_path)

else:
    baseline_pipeline.fit(df_train_es[features], df_train_es[task])
    y_pred_es = baseline_pipeline.predict(df_val_es[features])
    baseline_pipeline_en.fit(df_train_en[features], df_train_en[task])
    y_pred_en = baseline_pipeline_en.predict(df_val_en[features])

    unique_label = np.unique(df_train_es[task])

    print("Classification report:::")
    print(classification_report(
        pd.concat([df_val_es[task], df_val_en[task]]), 
        np.concatenate((y_pred_es, y_pred_en), axis=None), target_names=unique_label))
    print(f'Saving model at {model_path}')
    joblib.dump(baseline_pipeline, model_path)
    print(f'Saving model at {model_path_english}')
    joblib.dump(baseline_pipeline_en, model_path_english)


# print("Confusion matrix:::")
# confusion_matrix = pd.DataFrame(confusion_matrix(tweets_labeled[task], y_pred, labels=unique_label), index=['true:{:}'.format(x) for x in unique_label], columns=['pred:{:}'.format(x) for x in unique_label])
# print(confusion_matrix)
# confusion_matrix.to_csv('../data/output/confusion_matrix.csv', encoding='utf-8')
#tweets_labeled = tweets_labeled.assign(y_pred=pd.Series(y_pred).values)
#tweets_labeled.to_csv('corpus_y_pred.csv', sep = ';', encoding='utf-8')
end = time.time()
print("Process time:", (end - start)/60, "minutes")

#%% Método de evaluación: 
#1. Hacer 10 repartos diferentes y aleatorios de training y test (70/30) -> ShuffleSplit
#2. Para cada reparto, 
#	2.1 hacer GridSearchCV con el  70%, coger parámetros óptimos
#	2.2 hacer un cross_validate (k=10) con el 30% restante y los parámetros óptimos
#
#Problema: Los parámetros pueden cambiar en cada caso!?
########################
#import time
#start = time.time()
#scoring = {'acc': 'accuracy',
#           'precision': 'precision_macro',
#           'recall': 'recall_macro',
#           'f1': 'f1_macro'
#           }
#
#parameters = utils.get_grid_parameters(classifier)
#model = GridSearchCV(classifier_pipeline, param_grid=parameters, cv=5,
#                         scoring='f1_macro', verbose=1, n_jobs = -1)
#from sklearn.model_selection import ShuffleSplit
#
#scores_fold = []
#train_test_split = ShuffleSplit(n_splits=10, test_size=.70, random_state=0)
#iter_split = 1
#for train, test in train_test_split.split(tweets_labeled):
#    train = tweets_labeled.iloc[train]
#    test = tweets_labeled.iloc[test]
#    model.fit(train[x_cols2], train['categoria'])
#    test_score = dict(cross_validate(model.best_estimator_, test[x_cols2], test['categoria'], cv = 10, n_jobs = -1, scoring=scoring))
#    scores_fold.append(test_score)
#    print(test_score)
#    unique_label = np.unique(test['categoria'])
#    y_pred = cross_val_predict(model.best_estimator_, test[x_cols2], test['categoria'], cv=10, n_jobs = -1)
#    print("Matriz de confusion:::::")
#    print(pd.DataFrame(confusion_matrix(test['categoria'], y_pred, labels=unique_label), index=['true:{:}'.format(x) for x in unique_label], columns=['pred:{:}'.format(x) for x in unique_label]))
#    test = test.assign(y_pred=pd.Series(y_pred).values)
#    file_name = 'test_' + str(iter_split) + '.csv'
#    iter_split += 1
#    test.to_csv(file_name, sep = ';', encoding='utf-8')
#
#
#keys = set([key for i in scores_fold for key,value in i.iteritems()])
#scores = {}
#for j in keys:
#    means = [np.mean(i[j]) for i in scores_fold]
#    scores[j] = np.mean(means)
#
#print(scores)    
#
#end = time.time()
#print(end - start)
#############################