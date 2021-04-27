import pandas as pd
import torch
import numpy as np
import transformer.datasets as datasets
from torch.utils.data import DataLoader


def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.to(device)
    return model


def get_result_test(model, dataloader, device, task):
    model.eval()
    probas, true_labels, predictions = [], [], []
    if task == 'multitask':
        probas_task2, true_labels_task2, predictions_task2 = [], [], []
    with torch.no_grad():
      for batch in dataloader:
          data = batch[0]
          b_input_ids = data[0].squeeze()
          b_input_mask = data[1].squeeze()  
          b_labels = batch[1].squeeze() 
          b_input_ids = b_input_ids.to(device, dtype=torch.long)
          b_input_mask = b_input_mask.to(device, dtype=torch.long)
          b_labels = b_labels.to(device, dtype=torch.long)
          if task != 'multitask':
              logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
          else:
              logits_task1, logits_task2 = model(input_id=b_input_ids, token_type_id=None, mask_id=b_input_mask)
          if task != "multitask":
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            probas.append(logits)
            true_labels.append(label_ids)
          else:
            logits_task1 = logits_task1.detach().cpu().numpy()
            logits_task2 = logits_task2.detach().cpu().numpy()
            labels_ids = b_labels.to('cpu').numpy()
            probas.append(logits_task1)
            probas_task2.append(logits_task2)
            true_labels.append(labels_ids)
    if task == 'multitask':
        #Task1
        for i in range(len(true_labels)):
            pred_=np.argmax(probas[i], axis=1)
            predictions.append(pred_)
        ids = np.concatenate(true_labels).ravel()
        predictions = np.concatenate(predictions).ravel()
        #print("TRUE_LABELS::::::: ", true_labels)
        #print("PREDICTIONS:::::: ", (predictions))
        #Task2
        for i in range(len(true_labels)):
            pred_=np.argmax(probas_task2[i], axis=1)
            predictions_task2.append(pred_)
        predictions_task2 = np.concatenate(predictions_task2).ravel()
        #print("TRUE_LABELS::::::: ", true_labels)
        #print("PREDICTIONS:::::: ", (predictions))
        return [ids, predictions, predictions_task2]
    else:
        for i in range(len(true_labels)):
            pred_=np.argmax(probas[i], axis=1)
            predictions.append(pred_)
        ids = np.concatenate(true_labels).ravel()
        predictions = np.concatenate(predictions).ravel()
        return [ids, predictions]
        #print("TRUE_LABELS::::::: ", true_labels)
        #print("PREDICTIONS:::::: ", (predictions))

def generate_submission(model_path, basenet, device, test_path=None, output_path=None, task=1, batch_size=2, sample=True, text_cleaner=False):
    dataset = datasets.exist_2021(test_path, sample = sample, basenet = basenet, is_test = True, text_cleaner=text_cleaner)
    test_data_loader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=batch_size)
    model=load_model(model_path, device)
    if task != 'multitask':
        ids, predictions = get_result_test(model, test_data_loader, device, task)
    if task == 'multitask':
        ids, predictions,predictions_task2 = get_result_test(model, test_data_loader, device, task)

    df = pd.read_table(test_path, sep="\t", dtype={'id': 'str'})
    if sample:
        df=df.sample(frac=0.01, random_state=123)

    df['id_'] = ids
    df['predictions'] = predictions
    if task==1:
        df['category']=df['predictions'].map({ 0: 'non-sexist', 1: 'sexist'})
        df=df[['id', 'test_case', 'category']]
        df.to_csv(output_path, sep="\t", index=False)
    elif task==2:
        df['category']=df['predictions'].map({0: 'non-sexist', 1: 'ideological-inequality', 2: 'stereotyping-dominance', 3: 'objectification', 4: 'sexual-violence', 5: 'misogyny-non-sexual-violence'})
        df=df[['id', 'test_case', 'category']]
        df.to_csv(output_path, sep="\t", index=False)
    elif task=='multitask':
        df['category']=df['predictions'].map({0: 'non-sexist', 1: 'sexist'})
        df=df[['id', 'test_case', 'category']]
        df.to_csv(output_path, sep="\t", index=False)
        df['category']=predictions_task2
        df['category']=df['category'].map({0: 'non-sexist', 1: 'ideological-inequality', 2: 'stereotyping-dominance', 3: 'objectification', 4: 'sexual-violence', 5: 'misogyny-non-sexual-violence'})
        df.to_csv(output_path.replace('.tsv', '')+'_task2.tsv' , sep="\t", index=False)