import torch
import torch.nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np
import sys
import os

import sklearn
from sklearn import *
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_curve, auc
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from conf import *

import net as networks
import util

def get_optimizer(name, model, lr, wd):
    if name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f'Optimizer {name} is not supported yet.')

def get_lr(optimizer):
    for pg in list(optimizer.param_groups)[::-1]:
        return pg['lr']
        
# Function to save the model 
def saveModel(): 
    path = args.output
    torch.save(model.state_dict(), path) 
    
# Save Last Results into DataFrame
def saveResult(pred_epoch, true_epoch):
    pre_cat=np.concatenate((pred_epoch),axis=0).squeeze()
    true_cat=np.concatenate((true_epoch), axis=0).squeeze()
    result_df = pd.DataFrame({'pred':pre_cat.tolist(), 'label':true_cat.tolist()})
    path = args.exp_name + '.pkl'
    result_df.to_pickle(path)
    
def eval(model, loader):
    running_accuracy = 0
    running_vall_loss = 0
    pred_epoch = []
    true_epoch = []
    with torch.no_grad():
        model.eval()
        for data in loader:
            data = data.cuda()
            # indicies for taxonomy hyperedges' edge index
            out = model(data.x_n, data.edge_index_n, data.edge_mask, data.batch)  # Perform a single forward pass.
            pred_np = out.detach().cpu().numpy()
            true_np = data.y.unsqueeze(1).detach().cpu().numpy()
            val_loss = criterion(out, data.y.unsqueeze(1))  

            pred = out.argmax(dim=1)  # Use the class with highest probability.
            pred_epoch.append(pred_np)
            true_epoch.append(true_np)

            running_vall_loss += val_loss.item()
            running_accuracy += int((pred == data.y).sum())  # Check against ground-truth labels.

    # Calculate validation loss value
    val_loss_value = running_vall_loss/len(loader.dataset)

    accuracy = 100 * running_accuracy/len(loader.dataset)
    # Get AUPRC, AUROC
    true_values = np.concatenate((true_epoch), axis=0).squeeze().tolist()
    predicted_values = np.concatenate((pred_epoch),axis=0).squeeze().tolist()
    auprc = average_precision_score(true_values, predicted_values)
    auroc = roc_auc_score(true_values, predicted_values)

    return val_loss_value, accuracy, auprc, auroc, pred_epoch, true_epoch


# Training Function
def train(num_epochs, model, train_loader, val_loader,  test_loader, criterion):
    
    # for best summary
    best_accuracy = 0.0
    best_loss = 100000
    best_AUPRC = 0.0
    best_AUROC = 0.0    
    
    pred_list = []
     
    print("Begin training...") 
    step = 1
    for epoch in range(1, num_epochs+1): 
        running_train_loss = 0.0 

        total = 0 
         
        # Training Loop 
        print('training epoch',epoch)
        model.train()
        for data in tqdm(train_loader): 
            data = data.cuda()
            optimizer.zero_grad()   # zero the parameter gradients    
            
            out = model(data.x_n, data.edge_index_n, data.edge_mask, data.batch)  # Perform a single forward pass.
            loss = criterion(out.float(), data.y.unsqueeze(1).float()) # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            running_train_loss += loss.item()  # track the loss value 
            train_loss =  loss.item()/len(data.y)
            
            pred = out.argmax(dim=1)  # Use the class with highest probability.            
            running_train_accuracy = int((pred == data.y).sum())
            accuracy = 100 * running_train_accuracy/len(data.y)    
            step += 1
 
        # Calculate training loss value 
        train_loss_value = running_train_loss/len(train_loader.dataset) 
        val_loss, val_acc, val_auprc, val_auroc, _, _ = eval(model, val_loader)
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
        
        # get best loss
        if val_loss < best_loss:
            best_loss = val_loss
        
        # get best AUPRC
        # Save the model if the AUPRC is the best        
        if val_auprc > best_AUPRC:
            saveModel()             
            best_AUPRC = val_auprc
            
            # test model if the AUPRC is the best
            _, test_acc, test_auprc, test_auroc, test_pred_epoch, test_true_epoch = eval(model, test_loader)
            print('Epoch:',epoch,'========Test acc: %.4f,' %test_acc, 'Test auproc: %.4f,' %test_auprc, 'Test auroc: %.4f.' %test_auroc,'========')            
        
        # get best AUROC
        if val_auroc > best_AUROC:
            best_AUROC = val_auroc
        
        # Print the statistics of the epoch 
        print('Completed Epoch:', epoch, ', Training Loss : %.4f' %train_loss_value, ', Validation Loss : %.4f' %val_loss, ', Validation Accuracy : %.4f %%' %val_acc)
        
    print("========End Training========")
    _, test_acc, test_auprc, test_auroc, test_pred_epoch, test_true_epoch = eval(model, test_loader)
    print('Epoch:',epoch,'========Val acc: %.4f,' %val_acc, 'Val auproc: %.4f,' %val_auprc, 'Val auroc: %.4f.' %val_auroc,'========')            
    print('Epoch:',epoch,'========Test acc: %.4f,' %test_acc, 'Test auproc: %.4f,' %test_auprc, 'Test auroc: %.4f.' %test_auroc,'========')            
        
    return test_pred_epoch, test_true_epoch

if __name__ == '__main__':
    user_config()
    args = parse_arguments()    # from conf.py
    
    # Multi Hyperedge
    if args.dload == 'multi_hyper':
        sys.path.append('./graph_construction/')
        from LoadData import * #LoadPaitentData
        print('LoadData => LoadPaitentData imported')
    else:
        raise ValueError(f'Unknown dataloader: {args.dload}')
        
    util.seed_everything(args.seed)
    loader = LoadPaitentData(name=args.task, type='cutoff', tokenizer=args.tokenizer, data_type=args.dtype) 
    train_loader, val_loader, test_loader, n_class = loader.get_train_test(batch_size=args.bsz, seed=args.seed)
    
    # Define model            
    model = getattr(networks, args.model)(num_features=args.node_features, hidden_channels=args.hidden_channels*args.heads_1)        

    model.cuda() # Assume that GPU is available
    
    # Optimization settings
    if args.loss == 'BCEWithLogitsLoss':
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f'Unknown loss name: {args.loss}')
        
    optimizer = get_optimizer(args.optimizer, model, args.init_lr, args.weight_decay)

    # train model
    predicted_last, true_last = train(args.num_epochs, model, train_loader, val_loader, test_loader, criterion)

    # save last best results (from test data)
    saveResult(predicted_last, true_last)