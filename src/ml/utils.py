# -*- coding: utf-8 -*-
import numpy as np
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk import PorterStemmer
import re
#import emoji
import os
from nltk.tokenize import TweetTokenizer
import unidecode
from nltk.stem import SnowballStemmer
from sklearn.utils import resample
from sklearn.decomposition import PCA
from ml.preprocess import TextCleaner
import itertools


def to_lower_endline(text):
    return text.lower().replace('\n',' ')

def change_dtypes(df, convert_dict):
    return df.astype(convert_dict)

def remove_punctuation(text):
    return text.translate(None, string.punctuation)

def remove_accents(text):
    return unidecode.unidecode(text)

def replace_user(text):
    return re.sub( r"(?<=^|(?<=[^a-zA-Z0-9-\\.]))@([A-Za-z]+[A-Za-z0-9]+)", r"twuser", text)

def replace_hastags(text):
    return re.sub( r"(?<=^|(?<=[^a-zA-Z0-9-\\.]))#([A-Za-z]+[A-Za-z0-9]+)", r"twhastag", text)

def convert_hastags(text):
    return re.sub( r"([A-Z])", r" \1", text)

def replace_exclamation(text):
    return re.sub( r"(!+|¡+)", r" twexclamation", text)

def replace_url(text):
    return re.sub( r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", r"twurl", text)

def replace_interrogation(text):
    text = text.replace(u'¿', 'twinterrogation ')
    return re.sub( r"(\?+)", r" twinterrogation", text)

def replace_emoji(text):
    return emoji.demojize(unicode(text))

#Requieren tokenización:
def filter_stopwords(word_list):
    filtered_words = [word for word in word_list if word not in stopwords.words('spanish')]
    return filtered_words

def stemming(word_list):
    filtered_words_stem=[PorterStemmer().stem(word) for word in word_list]
    #filtered_words_stem=[SnowballStemmer('spanish').stem(word) for word in word_list]
    return filtered_words_stem

def replace_abb(tokens):
    path = os.getcwd()
    columns = ["word","label"]
    slang = pd.read_table(path + '/resources/lexicon/SP/SPslang.txt', names=columns, header=None, index_col=False)
    slang_dict = slang.set_index('word')['label'].to_dict()
    rep = dict((re.escape(k), v) for k, v in slang_dict.items())
    return replace(tokens, rep)

def replace(list, dictionary):
    new_list = []

    for i in list:
        if i in dictionary:
            new_list.append(dictionary[i])
        else:
          new_list.append(i)

    return new_list
          

def tokenizer_(text):
    tokens = TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(
            unidecode.unidecode(unidecode.unidecode(text)))
    #tokens = stemming(replace_abb(filter_stopwords(tokens)))
    tokens = filter_stopwords(tokens)
    return tokens

def tokenizer_v2(text):
    tokens = TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(
            unidecode.unidecode(unidecode.unidecode(text)))
    tokens = filter_stopwords(tokens)
    return tokens

def tokenizer_sin_stemming(text):
    tokens = TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(
            unidecode.unidecode(unidecode.unidecode(text.rstrip())))
    tokens = replace_abb(filter_stopwords(tokens))
    for token in tokens:
        if ('u00' in token) or (len(token) == 1) or token.isdigit():
            tokens.remove(token)
    return tokens

def replace_hurtlex(tokens):
    path = os.getcwd()
    columns = ["category","stereotype", "word"]
    hurtlex_conservative = pd.read_table(path + '/resources/hurtlex/hurtlex/hurtlex_ES_conservative.tsv', names=columns, header=None, index_col=False)
    hurtlex_conservative = hurtlex_conservative.loc[(hurtlex_conservative['category'] == 'asf') | (hurtlex_conservative['category'] <= 'pr')]
    hurtlex_conservative = hurtlex_conservative['word']
    hurtlex_inclusive = pd.read_table(path + '/resources/hurtlex/hurtlex/hurtlex_ES_inclusive.tsv', names=columns, header=None, index_col=False)
    hurtlex_inclusive = hurtlex_inclusive.loc[(hurtlex_inclusive['category'] == 'asf') | (hurtlex_inclusive['category'] <= 'pr')]
    hurtlex_inclusive = hurtlex_inclusive['word']
    hurtlex = set(hurtlex_conservative + hurtlex_inclusive)
    return(list(set(tokens) & set(hurtlex)))


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
    

def is_in_hurtlex_lexicon(text):
    path = os.getcwd()
    hurtlex_lexicon = pd.read_table(path + '/resources/hurtlex-29022020/lexica/ES/1.2/hurtlex_ES.tsv', index_col=False)
    hurtlex_lexicon_filtered = hurtlex_lexicon.loc[(hurtlex_lexicon['level'] == 'conservative')]
    hurtlex_lexicon_filtered = hurtlex_lexicon_filtered.loc[(hurtlex_lexicon['category'] == 'pa') | 
            (hurtlex_lexicon['category'] == 'dmc') | 
            (hurtlex_lexicon['category'] == 'asm') |
            (hurtlex_lexicon['category'] == 'asf') |
            (hurtlex_lexicon['category'] == 'pr') |
            (hurtlex_lexicon['category'] == 'om') |
            (hurtlex_lexicon['category'] == 'qas') |
            (hurtlex_lexicon['category'] == 'cds')]
    hurtlex_lexicon_filtered_lemma = hurtlex_lexicon['lemma']
    hurtlex_lemmas = list(set(hurtlex_lexicon_filtered_lemma))
    patts = re.compile("|".join(r"\b{}\b".format(s) for s in hurtlex_lemmas), re.I)
    tokens_in_lexicon = patts.findall(text)
    if len(tokens_in_lexicon) > 0:
        return(1)
    else:
        return(0)


def is_in_polar_lexicon(text):
    path = os.getcwd()
    polar_lexicon = pd.read_table(path + '/resources/polar/SentiSensePolares_ES.dat', index_col=False)
    polar_lexicon_lemma = polar_lexicon['Words']
    polar_lemmas = list(set(polar_lexicon_lemma))
    patts = re.compile("|".join(r"\b{}\b".format(s) for s in polar_lemmas), re.I)
    tokens_in_lexicon = patts.findall(text)
    if len(tokens_in_lexicon) > 0:
        return(1)
    else:
        return(0)

def n_in_hurtlex_lexicon(text, tokens_no_lexicon = False, filtered_sexism = True):
    path = os.getcwd()
    hurtlex_lexicon = pd.read_table(path + '/resources/hurtlex-29022020/lexica/ES/1.2/hurtlex_ES.tsv', index_col=False)
    hurtlex_lexicon_filtered = hurtlex_lexicon.loc[(hurtlex_lexicon['level'] == 'conservative')]
    hurtlex_lexicon_filtered = hurtlex_lexicon_filtered.loc[(hurtlex_lexicon['category'] == 'pa') | 
            (hurtlex_lexicon['category'] == 'dmc') | 
            (hurtlex_lexicon['category'] == 'asm') |
            (hurtlex_lexicon['category'] == 'asf') |
            (hurtlex_lexicon['category'] == 'pr') |
            (hurtlex_lexicon['category'] == 'om') |
            (hurtlex_lexicon['category'] == 'qas') |
            (hurtlex_lexicon['category'] == 'cds')]
    if filtered_sexism:
        hurtlex_lexicon_filtered_lemma = hurtlex_lexicon_filtered['lemma']
    else:
        hurtlex_lexicon_filtered_lemma = hurtlex_lexicon['lemma']
    hurtlex_lemmas = list(set(hurtlex_lexicon_filtered_lemma))
    patts = re.compile("|".join(r"\b{}\b".format(s) for s in hurtlex_lemmas), re.I)
    tokens_in_lexicon = patts.findall(text)
    count = len(tokens_in_lexicon)
    num_tokens = len(re.findall(r'\w+', text))
    num_no_lexicon = num_tokens - count
    if tokens_no_lexicon:
        return(count, num_no_lexicon)
    else:
        return(count)
        

def is_in_hatebase(text):
    path = os.getcwd()
    hatebase_english_lexicon = pd.read_csv(path + '/resources/hatebase/hatebase-english-vocab-translated.csv')
    hatebase_spanish_lexicon = pd.read_csv(path + '/resources/hatebase/hatebase-spanish-vocab.csv')
    hatebase_lemmas = list(set(hatebase_english_lexicon['term']) | set(hatebase_spanish_lexicon['term']))
    patts = re.compile("|".join(r"\b{}\b".format(s) for s in hatebase_lemmas), re.I)
    tokens_in_lexicon = patts.findall(text)
    if len(tokens_in_lexicon) > 0:
        return(1)
    else:
        return(0)


def is_in_metwo_lexicon(text):
    path = os.getcwd()
    metwo_lexicon = pd.read_csv(path + '/resources/metwo_lexicon/metwo_lexicon_k10_words_150.csv')
    metwo_lemmas = metwo_lexicon['term']
    #message = 'eres muy tortillera'
    preprocessor = TextCleaner(filter_users=True, filter_hashtags=True, 
                           filter_urls=True, convert_hastags=True, lowercase=True, 
                           replace_exclamation=True, replace_interrogation=True, 
                           remove_accents=True, remove_punctuation=True, replace_emojis=True) 
    tokens_text = tokenizer_sin_stemming(preprocessor(text))
    text = ' '.join(tokens_text)
    patts = re.compile("|".join(r"\b{}\b".format(s) for s in metwo_lemmas), re.I)
    tokens_in_lexicon = patts.findall(text)
    if len(tokens_in_lexicon) > 0:
        return(1)
    else:
        return(0)


def is_in_metwo_lexicon_v2(text):
    path = os.getcwd()
    metwo_lexicon = pd.read_table(path + '/resources/metwo_lexicon/metwo_lexicon_v2_k10_words_100.csv', index_col=False)
    metwo_lemmas = metwo_lexicon['term']
    preprocessor = TextCleaner(filter_users=False, filter_hashtags=False, 
                               filter_urls=True, convert_hastags=False, lowercase=True, 
                               replace_exclamation=False, replace_interrogation=False, 
                               remove_accents=False, remove_punctuation=False, replace_emojis=True) 
    text = preprocessor(text)
    patts = re.compile("|".join(r"\b{}\b".format(s) for s in metwo_lemmas), re.I)
    tokens_in_lexicon = patts.findall(text)
    if len(tokens_in_lexicon) > 0:
        return(1)
    else:
        return(0)


def n_in_hatebase(text, tokens_no_lexicon = False, filtered_sexism = True):
    path = os.getcwd()
    if filtered_sexism:
        hatebase_english_lexicon = pd.read_csv(path + '/resources/hatebase/hatebase-english-vocab-translated.csv')
        hatebase_spanish_lexicon = pd.read_csv(path + '/resources/hatebase/hatebase-spanish-vocab.csv')
    else:
        hatebase_english_lexicon = pd.read_csv(path + '/resources/hatebase/hatebase-english-all-vocab-translated.csv')
        hatebase_spanish_lexicon = pd.read_csv(path + '/resources/hatebase/hatebase-spanish-all-vocab.csv')
    hatebase_lemmas = list(set(hatebase_english_lexicon['term']) | set(hatebase_spanish_lexicon['term']))
    patts = re.compile("|".join(r"\b{}\b".format(s) for s in hatebase_lemmas), re.I)
    tokens_in_lexicon = patts.findall(text)
    count = len(tokens_in_lexicon)
    num_tokens = len(re.findall(r'\w+', text))
    num_no_lexicon = num_tokens - count
    if tokens_no_lexicon:
        return(count, num_no_lexicon)
    else:
        return(count)


def n_in_metwo_lexicon(text, tokens_no_lexicon = False):
    path = os.getcwd()
    metwo_lexicon = pd.read_csv(path + '/resources/metwo_lexicon/metwo_lexicon_k10_words_100.csv')
    metwo_lemmas = metwo_lexicon['term']
    #message = 'eres muy tortillera'
    preprocessor = TextCleaner(filter_users=True, filter_hashtags=True, 
                           filter_urls=True, convert_hastags=True, lowercase=True, 
                           replace_exclamation=True, replace_interrogation=True, 
                           remove_accents=True, remove_punctuation=True, replace_emojis=True) 
    tokens_text = tokenizer_sin_stemming(preprocessor(text))
    text = ' '.join(tokens_text)
    patts = re.compile("|".join(r"\b{}\b".format(s) for s in metwo_lemmas), re.I)
    tokens_in_lexicon = patts.findall(text)
    count = len(tokens_in_lexicon)
    num_tokens = len(re.findall(r'\w+', text))
    num_no_lexicon = num_tokens - count
    if tokens_no_lexicon:
        return(count, num_no_lexicon)
    else:
        return(count)


def n_in_metwo_lexicon_v2(text, tokens_no_lexicon = False):
    path = os.getcwd()
    metwo_lexicon = pd.read_table(path + '/resources/metwo_lexicon/metwo_lexicon_v2_k10_words_250_manual_inspection.csv', index_col=False)
    metwo_lemmas = metwo_lexicon['term']
    preprocessor = TextCleaner(filter_users=False, filter_hashtags=False, 
                               filter_urls=True, convert_hastags=False, lowercase=True, 
                               replace_exclamation=False, replace_interrogation=False, 
                               remove_accents=False, remove_punctuation=False, replace_emojis=True) 
    text = preprocessor(text)
    patts = re.compile("|".join(r"\b{}\b".format(s) for s in metwo_lemmas), re.I)
    tokens_in_lexicon = patts.findall(text)
    count = len(tokens_in_lexicon)
    num_tokens = len(re.findall(r'\w+', text))
    num_no_lexicon = num_tokens - count
    if tokens_no_lexicon:
        return(count, num_no_lexicon)
    else:
        return(count)


def n_in_polar_lexicon(text, tokens_no_lexicon = False):
    path = os.getcwd()
    polar_lexicon = pd.read_table(path + '/resources/polar/SentiSensePolares_ES.dat', index_col=False)
    polar_lexicon_lemma = polar_lexicon['Words']
    polar_lemmas = list(set(polar_lexicon_lemma))
    patts = re.compile("|".join(r"\b{}\b".format(s) for s in polar_lemmas), re.I)
    tokens_in_lexicon = patts.findall(text)
    count = len(tokens_in_lexicon)
    num_tokens = len(re.findall(r'\w+', text))
    num_no_lexicon = num_tokens - count
    if tokens_no_lexicon:
        return(count, num_no_lexicon)
    else:
        return(count)



def print_topics(model, count_vectorizer, n_top_words, tf_feature_names):
    words = tf_feature_names
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


def get_n_words_from_topic(model, n_top_words, tf_feature_names, topics):
    lexicon = []
    words = tf_feature_names
    for topic_idx, topic in enumerate(model.components_):
        if topic_idx in topics:
            #print(topic_idx)
            words_topic = [words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            #print(words_topic)
            lexicon += words_topic
    return(lexicon)


def tokens_in_hatebase(text, filtered_sexism = True):
    path = os.getcwd()
    if filtered_sexism:
        hatebase_english_lexicon = pd.read_csv(path + '/resources/hatebase/hatebase-english-vocab-translated.csv')
        hatebase_spanish_lexicon = pd.read_csv(path + '/resources/hatebase/hatebase-spanish-vocab.csv')
    else:
        hatebase_english_lexicon = pd.read_csv(path + '/resources/hatebase/hatebase-english-all-vocab-translated.csv')
        hatebase_spanish_lexicon = pd.read_csv(path + '/resources/hatebase/hatebase-spanish-all-vocab.csv')
    hatebase_lemmas = list(set(hatebase_english_lexicon['term']) | set(hatebase_spanish_lexicon['term']))
    tokens = [0]*len(hatebase_lemmas)
    hatebase_lemmas = [x.lower() for x in hatebase_lemmas]
    patts = re.compile("|".join(r"\b{}\b".format(s) for s in hatebase_lemmas), re.I)
    tokens_in_lexicon = patts.findall(text)
    if len(tokens_in_lexicon)>0:
        for x in tokens_in_lexicon:
            tokens[hatebase_lemmas.index(x.lower())]=1
    return(tokens)


def tokens_in_hurtlex_lexicon(text, filtered_sexism = True):
    path = os.getcwd()
    hurtlex_lexicon = pd.read_table(path + '/resources/hurtlex-29022020/lexica/ES/1.2/hurtlex_ES.tsv', index_col=False)
    hurtlex_lexicon_filtered = hurtlex_lexicon.loc[(hurtlex_lexicon['level'] == 'conservative')]
    hurtlex_lexicon_filtered = hurtlex_lexicon_filtered.loc[(hurtlex_lexicon['category'] == 'pa') | 
            (hurtlex_lexicon['category'] == 'dmc') | 
            (hurtlex_lexicon['category'] == 'asm') |
            (hurtlex_lexicon['category'] == 'asf') |
            (hurtlex_lexicon['category'] == 'pr') |
            (hurtlex_lexicon['category'] == 'om') |
            (hurtlex_lexicon['category'] == 'qas') |
            (hurtlex_lexicon['category'] == 'cds')]
    if filtered_sexism:
        hurtlex_lexicon_filtered_lemma = hurtlex_lexicon_filtered['lemma']
    else:
        hurtlex_lexicon_filtered_lemma = hurtlex_lexicon['lemma']
    hurtlex_lemmas = list(set(hurtlex_lexicon_filtered_lemma))
    tokens = [0]*len(hurtlex_lemmas)
    patts = re.compile("|".join(r"\b{}\b".format(s) for s in hurtlex_lemmas), re.I)
    tokens_in_lexicon = patts.findall(text)
    if len(tokens_in_lexicon)>0:
        for x in tokens_in_lexicon:
            tokens[hurtlex_lemmas.index(x.lower())]=1
    return(tokens)


def tokens_in_metwo_lexicon(text):
    path = os.getcwd()
    metwo_lexicon = pd.read_csv(path + '/resources/metwo_lexicon/disjoint_machista.csv')
    metwo_lemmas = list(metwo_lexicon['term'])
    #message = 'eres muy tortillera'
    preprocessor = TextCleaner(filter_users=True, filter_hashtags=True, 
                           filter_urls=True, convert_hastags=True, lowercase=True, 
                           replace_exclamation=True, replace_interrogation=True, 
                           remove_accents=True, remove_punctuation=True, replace_emojis=True) 
    tokens_text = tokenizer_sin_stemming(preprocessor(text))
    tokens = [0]*len(metwo_lemmas)
    text = ' '.join(tokens_text)
    patts = re.compile("|".join(r"\b{}\b".format(s) for s in metwo_lemmas), re.I)
    tokens_in_lexicon = patts.findall(text)
    if len(tokens_in_lexicon)>0:
        for x in tokens_in_lexicon:
            tokens[metwo_lemmas.index(x)]=1
    return(tokens)


def tokens_in_metwo_lexicon_v2(text, tokens_no_lexicon = False):
    path = os.getcwd()
    metwo_lexicon = pd.read_table(path + '/resources/metwo_lexicon/metwo_lexicon_v2_k10_words_100.csv', index_col=False)
    metwo_lemmas = list(metwo_lexicon['term'])
    preprocessor = TextCleaner(filter_users=False, filter_hashtags=False, 
                               filter_urls=True, convert_hastags=False, lowercase=True, 
                               replace_exclamation=False, replace_interrogation=False, 
                               remove_accents=False, remove_punctuation=False, replace_emojis=True) 
    text = preprocessor(text)
    patts = re.compile("|".join(r"\b{}\b".format(s) for s in metwo_lemmas), re.I)
    tokens_in_lexicon = patts.findall(text)
    tokens = [0]*len(metwo_lemmas)
    if len(tokens_in_lexicon)>0:
        for x in tokens_in_lexicon:
            tokens[metwo_lemmas.index(x.lower())]=1
    return(tokens)

def tokens_in_polar_lexicon(text, filtered_sexism = True):
    path = os.getcwd()
    polar_lexicon = pd.read_table(path + '/resources/polar/SentiSensePolares_ES.dat', index_col=False)
    polar_lexicon_lemma = polar_lexicon['Words']
    polar_lemmas = list(set(polar_lexicon_lemma))
    polar_lemmas = [x.lower() for x in polar_lemmas]
    tokens = [0]*len(polar_lemmas)
    patts = re.compile("|".join(r"\b{}\b".format(s) for s in polar_lemmas), re.I)
    tokens_in_lexicon = patts.findall(text)
    if len(tokens_in_lexicon)>0:
        for x in tokens_in_lexicon:
            tokens[polar_lemmas.index(x.lower())]=1
    return(tokens)



def add_vector_lexicon(lexicon, df, pca_components = 0):
    lexicon_vector=[]
    if lexicon == 'hurtlex':
        for index, row in df.iterrows():
            lexicon_vector.append(tokens_in_hurtlex_lexicon(row['text']))
    elif lexicon == 'metwo':
        print('Ejecutando MeTwo:')
        for index, row in df.iterrows():
            lexicon_vector.append(tokens_in_metwo_lexicon_v2(row['text']))
    elif lexicon == 'polar':
        for index, row in df.iterrows():
            lexicon_vector.append(tokens_in_polar_lexicon(row['text']))        
    else:
        for index, row in df.iterrows():
            lexicon_vector.append(tokens_in_hatebase(row['text']))
    lexicon_df = pd.DataFrame(lexicon_vector)
    if pca_components != 0:
        pca = PCA(n_components=pca_components)
        pca_data = lexicon_df.values
        principalComponents = pca.fit_transform(pca_data)
        lexicon_df = pd.DataFrame(data = principalComponents)
    tweets_lexicon_df = pd.concat([df, lexicon_df], axis=1)
    return(lexicon_df.columns.tolist(),tweets_lexicon_df)


def add_vector_lexicon_v2(lexicon, df, pca_components = 0):
    lexicon_vector=[]
    if lexicon == 'hurtlex':
        for index, row in df.iterrows():
            lexicon_vector.append(tokens_in_hurtlex_lexicon(row['text']))
    elif lexicon == 'metwo':
        print('Ejecutando tokens_in_metwo_lexicon_v2: ')
        for index, row in df.iterrows():
            lexicon_vector.append(tokens_in_metwo_lexicon_v2(row['text']))
    elif lexicon == 'polar':
        for index, row in df.iterrows():
            lexicon_vector.append(tokens_in_polar_lexicon(row['text']))        
    else:
        for index, row in df.iterrows():
            lexicon_vector.append(tokens_in_hatebase(row['text']))
    lexicon_df_ = pd.DataFrame(lexicon_vector)
    lexicon_df_inv = (lexicon_df_/lexicon_df_.sum(axis=0))*len(lexicon_df_)
    lexicon_df = lexicon_df_inv.replace(np.nan, 0)
    if pca_components != 0:
        pca = PCA(n_components=pca_components)
        pca_data = lexicon_df.values
        principalComponents = pca.fit_transform(pca_data)
        lexicon_df = pd.DataFrame(data = principalComponents)
    tweets_lexicon_df = pd.concat([df, lexicon_df], axis=1)
    return(lexicon_df.columns.tolist(),tweets_lexicon_df)


def token_document_frequency(tweets_):
    d = {el:0 for el in list(set(list(itertools.chain(*tweets_.tolist()))))}
    tweets = tweets_.tolist()
    for term in d.keys():
        for tweet in tweets:
            if(any(term == c for c in tweet)):
                d[term] += 1
        d[term] = d[term]/len(tweets_)
    df_ = pd.DataFrame(d.items(), columns=['term', 'value']).sort_values('value', ascending=False).reset_index(drop=True)
    return (df_)
    

def get_grid_parameters(method='logistic_regression'):
    if 'logistic_regression' == method:
        return {
#    'feature-union__text-features__tfidf__use_idf': (True, False),
#    'feature-union__text-features__tfidf__smooth_idf': (True, False),
#    'feature-union__text-features__tfidf__sublinear_tf': (True, False),
#    'feature-union__text-features__tfidf__norm': ['l2', None],
#    'feature-union__text-features__tfidf__min_df': [0.001, 0.01, 1],
    'clf__C': [1, 10],
    'clf__multi_class': ['ovr', 'auto'],
    'clf__class_weight': [None, 'balanced']
}

    if 'random_forest' == method:
        return {
#    'feature-union__text-features__tfidf__use_idf': (True, False),
#    'feature-union__text-features__tfidf__smooth_idf': (True, False),
#    'feature-union__text-features__tfidf__sublinear_tf': (True, False),
#    'feature-union__text-features__tfidf__norm': ['l2', None],
#    'feature-union__text-features__tfidf__min_df': [0.0, 0.01, 1],
    'clf__class_weight': [None, 'balanced'],
    "clf__n_estimators": [250, 450],
    'clf__bootstrap': (True, False),
    'clf__max_depth': [None, 30]

}           
    if 'svm' == method:
        return {
#    'feature-union__text-features__tfidf__use_idf': (True, False),
#    'feature-union__text-features__tfidf__smooth_idf': (True, False),
#    'feature-union__text-features__tfidf__sublinear_tf': (True, False),
#    'feature-union__text-features__tfidf__norm': ['l2', None],
#    'feature-union__text-features__tfidf__min_df': [0.01, 1],
    'clf__class_weight': [None, 'balanced'],
    "clf__gamma": [0.001, 0.1, 0.6, 'auto'],
#    'clf__kernel': ['rbf'],
#    'clf__kernel': ['linear', 'rbf'],
    'clf__C': [1, 10, 100, 10000]
#    'clf__C': [1]
    
}
    if 'baseline' == method:
        return {
    'clf__C': [1, 10],
    'clf__multi_class': ['ovr', 'auto'],
    'clf__class_weight': [None, 'balanced']
}