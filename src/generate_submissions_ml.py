#%%
test_path = '../data/input/EXIST2021_dataset-test/EXIST2021_dataset/test/EXIST2021_test.tsv'
language = "both"
text_cleaner = False
sample = False
task = "task2"
classifier = 'svm'
features = ['text']
model_path = '../models/ml_test.pkl'
if language == "both":
    model_path_english = model_path.replace('.pkl', '')+'_english.pkl' 

#%%
pipeline = joblib.load(model_path)
y_pred_test = pipeline.predict(df_val[features])
