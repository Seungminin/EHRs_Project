import torch
import numpy as np
from torch.nn.functional import softmax

from sklearn.metrics import *
from sklearn.metrics import balanced_accuracy_score

from pipeline.utils import InfIter
from tqdm import tqdm
import torch.nn.functional as F
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
import os

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.9, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probabilities of the true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def evaluate_model(model, loader, prefix, writer, global_step):
    model.eval()
    scores = []
    accuracies = []
    auroc = []
    auprc = []
    losses = []

    y_preds = []
    y_trues = []
    probs = []
    loss_fn = torch.nn.CrossEntropyLoss()
    for batch in loader:
        x, y, padding_mask, static = map(lambda x:x.cuda(), batch)
        x = x.float().transpose(2, 1).cuda()
        y_pred = model(x, padding_mask, static.float())
        losses.append(loss_fn(y_pred, y.long().cuda()).item())
        prob = torch.nn.functional.softmax(y_pred, dim = 1) 
        predictions = torch.argmax(prob, dim=1).cpu().numpy()  
        y_preds.append(predictions)
        probs.append(prob.detach().cpu().numpy())
        y_ = y.cpu().numpy()
        y_trues.append(y_)
        # print(np.sum(predictions))
    prediction = np.concatenate(y_preds, axis = 0)
    true = np.concatenate(y_trues, axis = 0)
    prob = np.concatenate(probs, axis = 0)
    if np.sum(np.isnan(prob)) >0:
        writer.add_scalars('Loss', {prefix: np.mean(losses)}, global_step=global_step)
        print('Warning: Prob contains NaN. Skipped validation')
        return 0, 0, 0

    false_pos_rate, true_pos_rate, _ = roc_curve(true, prob[:, 1])  # 1D scores needed
    prec, rec, _ = precision_recall_curve(true, prob[:, 1])
    auroc = auc(false_pos_rate, true_pos_rate)
    auprc = auc(rec, prec)
    f1 = f1_score(true, prediction)
    if writer != None:
        writer.add_scalars('F1_score', {prefix: f1}, global_step=global_step)
        writer.add_scalars('Accuracy', {prefix: accuracy_score(true, prediction)}, global_step=global_step)
        writer.add_scalars('AUROC', {prefix: auroc}, global_step=global_step)
        writer.add_scalars('AUPRC', {prefix: auprc}, global_step=global_step)
        writer.add_scalars('Loss', {prefix: np.mean(losses)}, global_step=global_step)
    model.train()
    return np.mean(losses), accuracy_score(true, prediction), auroc, auprc

def evaluate_event(model_, loader, prefix, writer, global_step, loss_fn):
    model = model_.eval()
    auroc = []
    auprc = []
    losses = []

    y_preds = []
    y_trues = []
    probs = []
    for event_time, event_type, mort, statics in loader:
        event_type = event_type.long()
        y_pred = model(event_type, event_time, statics.float())
        losses.append(loss_fn(y_pred, y.long().cuda()).item())
        prob = torch.nn.functional.softmax(y_pred, dim = 1)  
        predictions = torch.argmax(prob, dim=1).cpu().numpy()  
        y_preds.append(predictions)
        probs.append(prob.detach().cpu().numpy())
        y_ = mort.cpu().numpy()
        y_trues.append(y_)
        # print(np.sum(predictions))
    prediction = np.concatenate(y_preds, axis = 0)
    true = np.concatenate(y_trues, axis = 0)
    prob = np.concatenate(probs, axis = 0)
    if np.sum(np.isnan(prob)) >0:
        writer.add_scalars('Loss', {prefix: np.mean(losses)}, global_step=global_step)
        print('Warning: Prob contains NaN. Skipped validation')
        return 0, 0, 0

    false_pos_rate, true_pos_rate, _ = roc_curve(true, prob[:, 1])  
    prec, rec, _ = precision_recall_curve(true, prob[:, 1])
    auroc = auc(false_pos_rate, true_pos_rate)
    auprc = auc(rec, prec)
    f1 = f1_score(true, prediction)
    if writer != None:
        writer.add_scalars('F1_score', {prefix: f1}, global_step=global_step)
        writer.add_scalars('Accuracy', {prefix: accuracy_score(true, prediction)}, global_step=global_step)
        writer.add_scalars('AUROC', {prefix: auroc}, global_step=global_step)
        writer.add_scalars('AUPRC', {prefix: auprc}, global_step=global_step)
        writer.add_scalars('Loss', {prefix: np.mean(losses)}, global_step=global_step)
    model_.train()
    return np.mean(losses), accuracy_score(true, prediction), auroc, auprc

def evaluate_mixed(model_, loader, prefix, writer, global_step, loss_fn):
    model = model_.eval()
    losses = []
    y_preds = []
    y_trues = []
    probs = []

    for batch in loader:
        vital, padding_mask, event_time, event_type, mort, statics = map(lambda x:x.cuda(), batch)
        x = vital.float().transpose(2, 1).cuda()
        event_type = event_type.long()
        y_pred = model(event_time, event_type, x, padding_mask, statics.float())
        losses.append(loss_fn(y_pred, mort.long()).item())
        prob = torch.nn.functional.softmax(y_pred, dim=1)  
        predictions = torch.argmax(prob, dim=1).cpu().numpy()  
        y_preds.append(predictions)
        probs.append(prob.detach().cpu().numpy())
        y_ = mort.cpu().numpy()
        y_trues.append(y_)

    prediction = np.concatenate(y_preds, axis=0)
    true = np.concatenate(y_trues, axis=0)
    prob = np.concatenate(probs, axis=0)

    # Handle NaN probabilities
    if np.sum(np.isnan(prob)) > 0:
        writer.add_scalars('Loss', {prefix: np.mean(losses)}, global_step=global_step)
        print('Warning: Prob contains NaN. Skipped validation')
        return 0, 0, 0, 0

    # Metrics calculation
    false_pos_rate, true_pos_rate, _ = roc_curve(true, prob[:, 1])
    prec, rec, _ = precision_recall_curve(true, prob[:, 1])
    auroc = auc(false_pos_rate, true_pos_rate)
    auprc = auc(rec, prec)
    f1 = f1_score(true, prediction)
    weighted_acc = balanced_accuracy_score(true, prediction)

    # Log metrics
    if writer is not None:
        writer.add_scalars('F1_score', {prefix: f1}, global_step=global_step)
        writer.add_scalars('Weighted_Accuracy', {prefix: weighted_acc}, global_step=global_step)
        writer.add_scalars('AUROC', {prefix: auroc}, global_step=global_step)
        writer.add_scalars('AUPRC', {prefix: auprc}, global_step=global_step)
        writer.add_scalars('Loss', {prefix: np.mean(losses)}, global_step=global_step)

    model_.train()
    return np.mean(losses), accuracy_score(true, prediction), auroc, auprc, f1, weighted_acc

# TS Supervised run
def supervised_run(model: torch.nn.Module, loaders, writer, learning_rate, l2_coef, record_freq, early_stopping = False, total_epoch = 100, warmup = 0):
    train_loader, val_loader, test_loader = loaders
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

    cnt = 0
    lowest_loss = 1e10
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in (pbar := tqdm(range(total_epoch))):
        for batch in train_loader:
            x, target, padding_masks, statics = map(lambda x:x.cuda(), batch)
            optimizer.zero_grad()
            x = x.float().transpose(2, 1)
            output = model(x, padding_masks, statics.float())
            loss = loss_fn(output, target.long().cuda())
            l2_regularization = torch.zeros(1).cuda()
            for pname, param in model.linear.named_parameters():
                if 'weight' in pname:
                    l2_regularization += torch.norm(param, 2).cuda()
            loss += (l2_coef * l2_regularization).squeeze()
            loss.backward()
            optimizer.step()
        if epoch % record_freq == record_freq - 1 and epoch > warmup:
            train_scores = evaluate_model(model, train_loader, 'train', writer, epoch)
            test_scores = evaluate_model(model, test_loader, 'test', writer, epoch)
            loss, acc, roc, prc = evaluate_model(model, val_loader, 'val', writer, epoch)
            if loss < lowest_loss:
                lowest_loss = loss
                best_test_scores = test_scores[1:]
                cnt = 0
            else:
                cnt+=1
            if cnt*record_freq > 30 and early_stopping:
                print('early stopped')
                break
            pbar.set_description('ACC:{} ROC:{} PRC:{}'.format(best_test_scores[0], best_test_scores[1], best_test_scores[2]))
    return best_test_scores

def evaluate_los(model, loader, prefix, writer, global_step, regression = False):
    model.eval()
    auroc = []
    auprc = []

    y_preds = []
    y_trues = []
    probs = []
    for batch in loader:
        vital, padding_mask, event_time, event_type, mort, statics = map(lambda x:x.cuda(), batch)
        event_type = event_type.long()
        x = vital.float().transpose(2, 1)
        y_pred = model(event_time, event_type, x, padding_mask, statics.float())
        if regression:
            y_preds.append(y_pred.detach().cpu().numpy())
        else:
            predictions = torch.argmax(y_pred, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
            y_preds.append(predictions)
        y_ = mort.cpu().numpy()
        y_trues.append(y_)
    prediction = np.concatenate(y_preds, axis = 0)
    true = np.concatenate(y_trues, axis = 0)
    model.train()
    if regression:
        mad = mean_absolute_error(true, prediction)
        return mad
    else:
        kappa = cohen_kappa_score(true, prediction, weights = 'linear')
        return kappa

def LOS_finetune(model, loaders, writer, learning_rate, loss_fn, record_freq, total_epoch = 100, l2_coef = 0, regression = False):
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate, weight_decay=l2_coef)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.5)

    train_loader, val_loader, test_loader = loaders
    step = 0
    cnt = 0
    suffix = 'LOS'
    if regression:
        best_score = 99
    else:
        best_score = 0
    losses = []
    for epoch in (pbar := tqdm(range(total_epoch))):
        for batch in train_loader:
            vital, padding_mask, event_time, event_type, mort, statics = map(lambda x:x.cuda(), batch)
            optimizer.zero_grad()
            event_type = event_type.long()
            x = vital.float().transpose(2, 1)
            pred = model(event_time, event_type, x, padding_mask, statics.float())

            if regression:
                loss = F.mse_loss(pred.squeeze(), mort.float())
            else:
                loss = F.cross_entropy(pred, mort.long())
            loss.backward()
            losses.append(loss.item())

            optimizer.step()
        # if epoch % record_freq == record_freq - 1:
        val_score = evaluate_los(model, val_loader, 'val', writer, epoch, regression)
        test_score = evaluate_los(model, test_loader, 'val', writer, epoch, regression)
        if (regression and (val_score < best_score)) or ((not regression) and (val_score > best_score)):
            best_score = val_score
            best_test_score = test_score
            best_state_dict = deepcopy(model.state_dict())
        else:
            cnt += 1
        if cnt > 30:
            print('early stopped')
            torch.save(best_state_dict, './trained/{}/whole_{:.4f}.dict'.format(suffix, best_test_score))
            return
        pbar.set_description('LOSS: {:.5f} val: {:.5f} test: {:.5f} best: {:.5f}'.format(np.mean(losses), val_score, test_score, best_score))
        losses.clear()
        if epoch % 20 == 19:
            scheduler.step()
    torch.save(best_state_dict, './trained/{}/whole_{:.4f}.dict'.format(suffix, best_test_score))
    return 

def mixed_finetune_balanced(model, loaders, writer, learning_rate, loss_fn, record_freq, total_epoch=100, l2_coef=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_coef)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    train0_loader, train1_loader, val_loader, test_loader = loaders
    step = 0
    cnt = 0
    suffix = 'M3'
    save_dir = f'./trained/{suffix}'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    highest_auroc = 0
    best_state_dict = None

    for epoch in (pbar := tqdm(range(total_epoch), desc="Training Epochs")):
        train1_iter = InfIter(train1_loader)

        # Training Loop
        for batch0 in train0_loader:
            batch1 = next(train1_iter)
            vital0, padding_mask0, event_time0, event_type0, mort0, statics0 = map(lambda x: x.cuda(), batch0)
            vital1, padding_mask1, event_time1, event_type1, mort1, statics1 = map(lambda x: x.cuda(), batch1)

            # Combine class-balanced batches
            vital, padding_mask, event_time, event_type, mort, statics = [
                torch.cat(i, 0) for i in [
                    (vital0, vital1), (padding_mask0, padding_mask1),
                    (event_time0, event_time1), (event_type0, event_type1),
                    (mort0, mort1), (statics0, statics1)
                ]
            ]

            optimizer.zero_grad()
            event_type = event_type.long()
            x = vital.float().transpose(2, 1)
            pred = model(event_time, event_type, x, padding_mask, statics.float())
            loss = loss_fn(pred, mort.long())
            loss.backward()
            optimizer.step()

            step += 1  # Increment training step

        # Validation and Metrics Calculation
        if epoch % record_freq == record_freq - 1:
            loss, val_acc, val_roc, val_prc, val_f1, val_weighted_acc = evaluate_mixed(model, val_loader, 'val', writer, epoch, loss_fn)
            test_loss, test_acc, test_roc, test_prc, test_f1, test_weighted_acc = evaluate_mixed(model, test_loader, 'test', writer, epoch, loss_fn)

            # Save the best model based on AUROC
            if val_roc > highest_auroc:
                highest_auroc = val_roc
                best_state_dict = deepcopy(model.state_dict())
                best_val_scores = (val_acc, val_roc, val_prc, val_f1, val_weighted_acc)
                best_test_scores = (test_acc, test_roc, test_prc, test_f1, test_weighted_acc)
                cnt = 0
            else:
                cnt += 1

            # Early stopping
            if cnt * record_freq > 30:
                print("Early stopped")
                torch.save(best_state_dict, f'{save_dir}/best_model_{highest_auroc:.4f}.dict')
                return best_val_scores, best_test_scores, step

            pbar.set_description(f"Epoch {epoch}: ACC: {val_acc:.4f}, ROC: {val_roc:.4f}, PRC: {val_prc:.4f}, F1: {val_f1:.4f}, Weighted_Acc: {val_weighted_acc:.4f}")

        # Adjust learning rate
        if epoch % 20 == 19:
            scheduler.step()

    # Save the final best model
    torch.save(best_state_dict, f'{save_dir}/final_model_{highest_auroc:.4f}.dict')
    return best_val_scores, best_test_scores, step

def mixed_finetune_imbalanced(model, loaders, writer, learning_rate, record_freq, total_epoch=100, l2_coef=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_coef)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    train0_loader, train1_loader, val_loader, test_loader = loaders
    step = 0
    cnt = 0
    suffix = 'M3'
    save_dir = f'./trained/{suffix}'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Combine all labels to calculate class weights
    train_labels = []
    for batch in train0_loader:
        _, _, _, _, mort, _ = batch
        train_labels.append(mort)
    for batch in train1_loader:
        _, _, _, _, mort, _ = batch
        train_labels.append(mort)

    combined_labels = torch.cat(train_labels).cpu().numpy().astype(np.int64)

    # Calculate class weights
    class_counts = np.bincount(combined_labels)
    class_weights = len(combined_labels) / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).cuda()

    # Define CrossEntropyLoss with class weights
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    highest_auroc = 0
    best_state_dict = None

    for epoch in (pbar := tqdm(range(total_epoch), desc="Training Epochs")):
        train1_iter = InfIter(train1_loader)

        # Training Loop
        for batch0 in train0_loader:
            batch1 = next(train1_iter)
            vital0, padding_mask0, event_time0, event_type0, mort0, statics0 = map(lambda x: x.cuda(), batch0)
            vital1, padding_mask1, event_time1, event_type1, mort1, statics1 = map(lambda x: x.cuda(), batch1)

            # Combine class-balanced batches
            vital, padding_mask, event_time, event_type, mort, statics = [
                torch.cat(i, 0) for i in [
                    (vital0, vital1), (padding_mask0, padding_mask1),
                    (event_time0, event_time1), (event_type0, event_type1),
                    (mort0, mort1), (statics0, statics1)
                ]
            ]

            optimizer.zero_grad()
            event_type = event_type.long()
            x = vital.float().transpose(2, 1)
            pred = model(event_time, event_type, x, padding_mask, statics.float())
            loss = loss_fn(pred, mort.long())
            loss.backward()
            optimizer.step()

            step += 1  # Increment training step

        # Validation and Metrics Calculation
        if step % 10 == 0:
            loss, val_acc, val_roc, val_prc, val_f1, val_weighted_acc = evaluate_mixed(model, val_loader, 'val', writer, epoch, loss_fn)
            test_loss, test_acc, test_roc, test_prc, test_f1, test_weighted_acc = evaluate_mixed(model, test_loader, 'test', writer, epoch, loss_fn)

            # Save the best model based on AUROC
            if val_roc > highest_auroc:
                highest_auroc = val_roc
                best_state_dict = deepcopy(model.state_dict())
                best_val_scores = (val_acc, val_roc, val_prc, val_f1, val_weighted_acc)
                best_test_scores = (test_acc, test_roc, test_prc, test_f1, test_weighted_acc)
                cnt = 0
            else:
                cnt += 1

            # Early stopping
            if cnt * record_freq > 30:
                print("Early stopped")
                torch.save(best_state_dict, f'{save_dir}/best_model_{highest_auroc:.4f}.dict')
                return best_val_scores, best_test_scores, step

            pbar.set_description(f"Epoch {epoch}: ACC: {val_acc:.4f}, ROC: {val_roc:.4f}, PRC: {val_prc:.4f}, F1: {val_f1:.4f}, Weighted_Acc: {val_weighted_acc:.4f}")

        # Adjust learning rate
        if epoch % 20 == 19:
            scheduler.step()

    # Save the final best model
    torch.save(best_state_dict, f'{save_dir}/final_model_{highest_auroc:.4f}.dict')
    return best_val_scores, best_test_scores, step


def mixed_finetune_balanced_graph(model, loaders, writer, learning_rate, loss_fn, record_freq, total_epoch=400, l2_coef=0, count = 1):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_coef)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    train0_loader, train1_loader, val_loader, test_loader = loaders
    step = 0
    cnt = 0
    suffix = 'M3'
    save_dir = f'./trained/{suffix}'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    highest_auroc = 0
    best_state_dict = None
    
    train_losses, val_losses = [], []  # 기록할 손실 값
    train_f1_scores, val_f1_scores, test_f1_scores = [], [], []  # 기록할 F1-Score 값

    for epoch in (pbar := tqdm(range(total_epoch), desc="Training Epochs")):
        train1_iter = InfIter(train1_loader)

        # Training Loop
        for batch0 in train0_loader:
            batch1 = next(train1_iter)
            vital0, padding_mask0, event_time0, event_type0, mort0, statics0 = map(lambda x: x.cuda(), batch0)
            vital1, padding_mask1, event_time1, event_type1, mort1, statics1 = map(lambda x: x.cuda(), batch1)

            # Combine class-balanced batches
            vital, padding_mask, event_time, event_type, mort, statics = [
                torch.cat(i, 0) for i in [
                    (vital0, vital1), (padding_mask0, padding_mask1),
                    (event_time0, event_time1), (event_type0, event_type1),
                    (mort0, mort1), (statics0, statics1)
                ]
            ]

            optimizer.zero_grad()
            event_type = event_type.long()
            x = vital.float().transpose(2, 1)
            pred = model(event_time, event_type, x, padding_mask, statics.float())
            loss = loss_fn(pred, mort.long())
            loss.backward()
            optimizer.step()

            step += 1  # Increment training step

        # Validation and Metrics Calculation
        if epoch % record_freq == record_freq - 1:
            train_loss, train_acc, train_roc, train_prc, train_f1, train_weighted_acc = evaluate_mixed(model, train0_loader, 'train', writer, epoch, loss_fn)
            val_loss, val_acc, val_roc, val_prc, val_f1, val_weighted_acc = evaluate_mixed(model, val_loader, 'val', writer, epoch, loss_fn)

            # Save the metrics for visualization
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_f1_scores.append(train_f1)
            val_f1_scores.append(val_f1)

            # Save the best model based on AUROC
            if val_roc > highest_auroc:
                highest_auroc = val_roc
                best_state_dict = deepcopy(model.state_dict())
                best_val_scores = (val_acc, val_roc, val_prc, val_f1, val_weighted_acc)
                best_test_scores = (val_acc, val_roc, val_prc, val_f1, val_weighted_acc)
                cnt = 0
            else:
                cnt += 1

            # Early stopping
            if cnt * record_freq > 30:
                print("Early stopped")
                torch.save(best_state_dict, f'{save_dir}/best_model_{highest_auroc:.4f}.dict')
                break

            pbar.set_description(f"Epoch {epoch}: ACC: {val_acc:.4f}, ROC: {val_roc:.4f}, PRC: {val_prc:.4f}, F1: {val_f1:.4f}, Weighted_Acc: {val_weighted_acc:.4f}")

        # Adjust learning rate
        if epoch % 20 == 19:
            scheduler.step()
        test_loss, test_acc, test_roc, test_prc, test_f1, test_weighted_acc = evaluate_mixed(model, test_loader, 'test', writer, epoch, loss_fn)
        test_f1_scores.append(test_f1)
        
    # 그래프 저장 코드 추가
    plt.figure(figsize=(12, 6))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # F1-Score plot
    plt.subplot(1, 2, 2)
    plt.plot(range(len(test_f1_scores)), test_f1_scores, label='Test F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.title('Test F1-Score')
    plt.legend()

    plt.tight_layout()

    # 파일로 저장
    plt.savefig(f'training_metrics_{count}.png')  # 'training_metrics.png'에 그래프 저장
    print("Training metrics saved as 'training_metrics.png'")

    # Save the final best model
    torch.save(best_state_dict, f'{save_dir}/final_model_{highest_auroc:.4f}.dict')
    return best_val_scores, best_test_scores, step



def evaluate_pheno(model, loader, suffix, writer = None, global_step = 0):
    model.eval()
    preds = []
    labels = []
    losses = []
    for batch in loader:
        vital, padding_mask, event_time, event_type, pheno, statics = map(lambda x:x.cuda(), batch)
        event_type = event_type.long()
        x = vital.float().transpose(2, 1)
        pred = model(event_time, event_type, x, padding_mask, statics.float())

        loss = F.binary_cross_entropy_with_logits(pred, pheno)
        losses.append(loss.item())
        pred = torch.sigmoid(pred)
        preds.append(pred.detach().cpu().numpy())
        labels.append(pheno.cpu().numpy())

    preds = np.concatenate(preds, axis = 0)
    labels = np.concatenate(labels, axis = 0)
    micro = roc_auc_score(labels, preds, average='micro')
    macro = roc_auc_score(labels, preds, average='macro')
    if writer != None:
        writer.add_scalars('AUROC', {'{}_micro'.format(suffix): micro}, global_step=global_step)
        writer.add_scalars('AUROC', {'{}_macro'.format(suffix): macro}, global_step=global_step)
        writer.add_scalars('Loss', {suffix: np.mean(losses)}, global_step=global_step)
    model.train()
    pred_logits = np.array(preds > 0.5)
    # np.save('./data/pheno_cf.npy', multilabel_confusion_matrix(labels, pred_logits))
    np.savez('./data/pheno_score.npz', y_true = labels, y_pred = preds)
    return micro, macro, np.mean(losses)

def pheno_finetune(model, loaders, writer, learning_rate, record_freq, total_epoch = 100, l2_coef = 0):
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate, weight_decay=l2_coef)
    lowest_loss = 1e10
    train_loader, val_loader, test_loader = loaders
    cnt = 0
    best_scores = (0,0)
    for epoch in (pbar := tqdm(range(total_epoch))):
        for batch in train_loader:
            vital, padding_mask, event_time, event_type, pheno, statics = map(lambda x:x.cuda(), batch)
            optimizer.zero_grad()
            event_type = event_type.long()
            x = vital.float().transpose(2, 1)
            pred = model(event_time, event_type, x, padding_mask, statics.float())

            """ backward """
            # print(prediction.shape, mort.shape)
            # loss = loss_fn(pred, mort.long())
            loss = F.binary_cross_entropy_with_logits(pred, pheno)
            loss.backward()

            """ update parameters """
            optimizer.step()
        test_scores = evaluate_pheno(model, test_loader, 'test', writer, epoch)
        val_scores = evaluate_pheno(model, val_loader, 'val', writer, epoch)
        micro, macro, loss = val_scores
        if val_scores[0] > best_scores[0]:
            best_scores = val_scores[0], val_scores[1]
            best_test_scores = test_scores
            best_state_dict = deepcopy(model.state_dict())
            cnt = 0
        else:
            cnt += 1
            if cnt > 30:
                torch.save(best_state_dict, './trained/pheno/{:.4f}.dict'.format(best_test_scores[0]))
                # return best_val_scores, best_test_scores, step
                return
        pbar.set_description('Loss: {:.5f}, best micro: {:.5f}, macro: {:.5f}, current micro: {:.5f}, macro: {:.5f}'.format(loss, best_test_scores[0], best_test_scores[1], test_scores[0], test_scores[1]))
    torch.save(best_state_dict, './trained/pheno/{:.4f}.dict'.format(best_test_scores[0]))
    # return best_val_scores, best_test_scores, step