import torch
import torch.nn as nn
import torch.quantization
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import get_cosine_schedule_with_warmup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from abc import ABC, abstractmethod

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, criterion, optimizer, train_loader, scheduler=None):
    
    running_loss, all_preds, all_labels = 0.0, [], []
    model.train()
    
    for inputs, labels, _ in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if scheduler is not None: scheduler.step()

    all_preds = np.argmax(all_preds, axis=1)
    avg_loss = running_loss / len(train_loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    
    return avg_loss, acc, f1

def test(model, criterion, loader):

    val_loss, val_preds, val_labels = 0.0, [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            val_preds.extend(outputs.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_preds = np.argmax(val_preds, axis=1)
    avg_loss = val_loss / len(loader.dataset)
    acc = accuracy_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds, average="macro")
    
    return avg_loss, acc, f1, val_preds, val_labels

class BaseTrainer(ABC):
    @abstractmethod
    def fit(self, *args, **kwargs):
        pass
        
    def evaluate(self, model, loader, display, out_filepath):
        model.to(device)
        model.eval()

        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        val_loss, val_acc, val_f1, preds, labels = test(model, criterion, loader)
        report = classification_report(labels, preds, output_dict=True)
        report = pd.DataFrame(report).transpose()
        
        if out_filepath is not None: report.to_csv(out_filepath, index=False)
        if display: print(report)
        return preds
        
    def training_loop(self, model, criterion, optimizer, scheduler, train_loader, valid_loader, epochs, model_save_directory):
        cnn = True
        if model.__class__.__name__ == 'ViT16': cnn = False

        model.to(device)
        best_val_file, best_f1 = None, None

        for epoch in range(epochs):
            # train
            if cnn: train(model, criterion, optimizer, train_loader)
            if not cnn: train(model, criterion, optimizer, train_loader, scheduler)

            # Evaluate
            val_loss, val_acc, val_f1, _, _ = test(model, criterion, valid_loader)

            if best_f1 is None or best_f1 < val_f1:
                best_f1 = val_f1
                best_val_file = f"{model_save_directory}model_{epoch}.pt"
                torch.save(model.state_dict(), best_val_file)

            if cnn: scheduler.step(val_f1)
            print(f'Epoch: {epoch} | Val Acc: {val_acc:.4f} | Loss: {val_loss:.4f} | F1: {val_f1:.4f}')
            
        torch.save(model.state_dict(), Path(model_save_directory) / f"model_{epochs-1}.pt")
        print(best_val_file)
        return best_val_file
            

class CNNModelTrainer(BaseTrainer):
    def __init__(self):
        self.class_weights = None
    def fit(self, model, train_loader, valid_loader, model_save_directory, epochs=200, lr=1e-5, plot=False):
        
        model.to(device)
        g_train_loss, g_train_acc, g_train_f1, g_val_loss, g_val_acc, g_val_f1, best_val_file, best_f1 = [], [], [], [], [], [], None, None

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5, momentum=0.9, nesterov=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=5)

        return self.training_loop(
            model, criterion, optimizer, scheduler, train_loader, valid_loader, epochs, model_save_directory)


class TransModelTrainer(BaseTrainer):
    def __init__(self):
        self.class_weights = None
    def fit(self, model, train_loader, valid_loader, model_save_directory, epochs=100, lr=1e-5, patience=30):
        
        model.to(device)
        criterion = nn.CrossEntropyLoss()

        no_decay = ['bias', 'ln_1.weight', 'ln_2.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

        num_training_steps = len(train_loader) * epochs
        num_warmup_steps = int(0.05 * num_training_steps)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
        return self.training_loop(
            model, criterion, optimizer, scheduler, train_loader, valid_loader, epochs, model_save_directory)