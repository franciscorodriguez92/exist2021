import transformers
import torch
from tqdm import tqdm, trange
from sklearn.metrics import f1_score, accuracy_score
import numpy as np



def train_one_epoch(model, dataloader, optimizer, device, scheduler, task):
    model.train()
    tr_loss = 0
    nb_tr_steps = 0
    for step, batch in enumerate(dataloader):
        #b_input_ids, b_input_mask, b_labels = batch
        data = batch[0]
        b_input_ids = data[0].squeeze()
        b_input_mask = data[1].squeeze()
        b_labels = batch[1].squeeze()
        b_input_ids = b_input_ids.to(device, dtype=torch.long)
        b_input_mask = b_input_mask.to(device, dtype=torch.long)
        b_labels = b_labels.to(device, dtype=torch.long)
        optimizer.zero_grad()
        #print(b_input_ids[:,:maxlength].shape)
        #print(b_input_ids.shape)
        #print(b_input_ids[:,maxlength:].shape)
        #print("IDS::::::: ", b_input_ids.shape)
        #print("MASKS::::::: ", b_input_mask[1][0])
        #print("LABELS:::::: ", b_labels.shape)
        #print("LABELS:::::: ", b_labels)
        if task != 'multitask':
            loss = model(input_ids=b_input_ids, token_type_ids=None, attention_mask=b_input_mask,labels=b_labels[:,task])[0]
        else:
            loss = model(input_id=b_input_ids, token_type_id=None, mask_id=b_input_mask,labels=b_labels)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
        nb_tr_steps += 1
        if scheduler:
            scheduler.step()
    print("\n\nTrain loss: {}".format(tr_loss/nb_tr_steps))
    return tr_loss/nb_tr_steps

def evaluate(model, dataloader, device, task):
    
    model.eval()
    test_loss, nb_test_steps=0, 0

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
          # with torch.no_grad():
          #     logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
          if task != 'multitask':
              loss = model(input_ids=b_input_ids, token_type_ids=None, attention_mask=b_input_mask,labels=b_labels[:,task])[0]
              logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
          else:
              loss = model(input_id=b_input_ids, token_type_id=None, mask_id=b_input_mask,labels=b_labels)
              logits_task1, logits_task2 = model(input_id=b_input_ids, token_type_id=None, mask_id=b_input_mask)
          test_loss += loss.item()
          nb_test_steps += 1
          if task != "multitask":
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels[:,task].to('cpu').numpy()
            probas.append(logits)
            true_labels.append(label_ids)
          else:
            logits_task1 = logits_task1.detach().cpu().numpy()
            logits_task2 = logits_task2.detach().cpu().numpy()
            labels_ids_task1 = b_labels[:,0].to('cpu').numpy()
            labels_ids_task2 = b_labels[:,1].to('cpu').numpy()
            probas.append(logits_task1)
            probas_task2.append(logits_task2)
            true_labels.append(labels_ids_task1)
            true_labels_task2.append(labels_ids_task2)
    if task == 'multitask':
        #Task1
        for i in range(len(true_labels)):
            pred_=np.argmax(probas[i], axis=1)
            predictions.append(pred_)
        true_labels = np.concatenate(true_labels).ravel()
        predictions = np.concatenate(predictions).ravel()
        #print("TRUE_LABELS::::::: ", true_labels)
        #print("PREDICTIONS:::::: ", (predictions))
        test_acc = accuracy_score(true_labels, predictions)
        test_f_score = f1_score(true_labels, predictions, average='macro')
        print(f'\n\n Results Task 1 --- Test loss: {test_loss/nb_test_steps}, Test acc: {test_acc}, Test f1: {test_f_score}')
        #Task2
        for i in range(len(true_labels_task2)):
            pred_=np.argmax(probas_task2[i], axis=1)
            predictions_task2.append(pred_)
        true_labels_task2 = np.concatenate(true_labels_task2).ravel()
        predictions_task2 = np.concatenate(predictions_task2).ravel()
        #print("TRUE_LABELS::::::: ", true_labels)
        #print("PREDICTIONS:::::: ", (predictions))
        test_acc_task2 = accuracy_score(true_labels_task2, predictions_task2)
        test_f_score_task2 = f1_score(true_labels_task2, predictions_task2, average='macro')
        print(f'\n\n Results Task 1 --- Test loss: {test_loss/nb_test_steps}, Test acc: {test_acc_task2}, Test f1: {test_f_score_task2}')
    else:
        for i in range(len(true_labels)):
            pred_=np.argmax(probas[i], axis=1)
            predictions.append(pred_)
        true_labels = np.concatenate(true_labels).ravel()
        predictions = np.concatenate(predictions).ravel()
        #print("TRUE_LABELS::::::: ", true_labels)
        #print("PREDICTIONS:::::: ", (predictions))
        test_acc = accuracy_score(true_labels, predictions)
        test_f_score = f1_score(true_labels, predictions, average='macro')
        print(f'\n\nTest loss: {test_loss/nb_test_steps}, Test acc: {test_acc}, Test f1: {test_f_score}')       
    #return test_loss/nb_test_steps, [test_acc, test_f_score]
    return {'loss': test_loss/nb_test_steps, 'acc': test_acc, 'macro_f1': test_f_score}

def train(model, optimizer,
                train_loader, validation_loader, epochs, model_path, scheduler=None,
                monitor="f1", early_stopping_tolerance=5, task=1):
    if task != 'multitask':
        task = 0 if task==1 else 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs_without_improvement =0
    best_metrics = None
    def improves_performance(best_metrics, metrics):
        if best_metrics is None:
            return True

        if monitor == "loss":
            if metrics['loss'] < best_metrics['loss']:
                return True
            else:
                return False
        elif monitor == "f1":
            if metrics['macro_f1'] > best_metrics['macro_f1']:
                return True
            else:
                return False
    
    for _ in trange(epochs, desc="Epoch"):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scheduler, task)
        metrics = evaluate(model, validation_loader, device, task)
        if improves_performance(best_metrics, metrics):
            best_metrics = metrics
            epochs_without_improvement = 0
            torch.save(model, model_path)
            print(f"\n\nBest model so far ({best_metrics}) saved at {model_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_tolerance:
                print("Early stopping")
                break
    print("\n\nTraining process completed")
    

    
def train_and_predict_transformer(model, train_dataloader, test_dataloader, optimizer, scheduler, device, epochs = 5, task = 1):
    task = 0 if task==1 else 1
    for _ in trange(epochs, desc="Epoch"):
      model.train()
      tr_loss = 0
      nb_tr_examples, nb_tr_steps = 0, 0 
      for step, batch in enumerate(train_dataloader):
        #b_input_ids, b_input_mask, b_labels = batch
        data = batch[0]
        b_input_ids = data[0].squeeze()
        b_input_mask = data[1].squeeze()
        b_labels = batch[1][:,task].squeeze()
        b_input_ids = b_input_ids.to(device, dtype=torch.long)
        b_input_mask = b_input_mask.to(device, dtype=torch.long)
        b_labels = b_labels.to(device, dtype=torch.long)
        optimizer.zero_grad()
        #print(b_input_ids[:,:maxlength].shape)
        #print(b_input_ids.shape)
        #print(b_input_ids[:,maxlength:].shape)
        #print("IDS::::::: ", b_input_ids.shape)
        #print("MASKS::::::: ", b_input_mask[1][0])
        #print("LABELS:::::: ", b_labels.shape)
        #print("LABELS:::::: ", b_labels)
        loss = model(input_ids=b_input_ids, token_type_ids=None, attention_mask=b_input_mask,labels=b_labels)[0]
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
        nb_tr_steps += 1
        scheduler.step()
      print("Train loss: {}".format(tr_loss/nb_tr_steps))
    
    model.eval()

    predictions , true_labels = [], []

    for batch in test_dataloader:
      #b_input_ids, b_input_mask, b_labels = batch
      data = batch[0]
      b_input_ids = data[0].squeeze()
      b_input_mask = data[1].squeeze()  
      b_labels = batch[1][:,0].squeeze() 
      b_input_ids = b_input_ids.to(device, dtype=torch.long)
      b_input_mask = b_input_mask.to(device, dtype=torch.long)
      b_labels = b_labels.to(device, dtype=torch.long)
      with torch.no_grad():
        logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()
      predictions.append(logits)
      true_labels.append(label_ids)
    return predictions, true_labels