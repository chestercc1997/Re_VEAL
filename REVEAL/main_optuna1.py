import argparse
import torch
import optuna
from optuna.trial import TrialState
# import opentuna
# from opentuna import Param, optimize
import torch.utils.data as Data
from torch.utils.data import random_split
import torch_geometric.transforms as T
import torch_geometric
from logger import Logger
import os
import sys
import numpy as np
import pandas as pd
import copy
import torch
from torch.cuda.amp import autocast, GradScaler
from utils import *
from dataset_prep import PygNodePropPredDataset, Evaluator
from model import SynthNet,SimpleGNN1,SimpleGNN_SAGPool,SimpleGNN1_HOGA
from torch_geometric.data import Dataset, Data
from functools import partial
import os.path as osp
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import multiprocessing as mp
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import json
os.environ['TORCH_HOME'] = '/tmp/.cache/torch'


global best_top1_metrics
best_top1_metrics = (0.0, 0.0, 0.0)  # (Top-1, Top-2, Top-3)

global best_top1_model_state
best_top1_model_state = None

global best_top3_metrics
best_top3_metrics = (0.0, 0.0, 0.0)  # (Top-1, Top-2, Top-3)

global best_top3_model_state
best_top3_model_state = None


global best_top1 
best_top1 = 0.0
global best_top3
best_top3 = 0.0



class MyOwnDataset(Dataset):
    def __init__(self, root, args, transform=None, pre_transform=None, pre_filter=None):
        self.args = args
        
        self.root = root
        self.scaler_path = osp.join(self.root, 'graph_feature_scaler.pkl')
        super(MyOwnDataset, self).__init__(root, transform, pre_transform, pre_filter)
        if osp.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
        else:
            self.scaler = None

    @property
    def raw_file_names(self):
        return os.listdir(self.root)

    @property
    def processed_file_names(self):
        file_names = os.listdir(self.processed_dir)
        return [f for f in file_names if f not in ['pre_transform.pt', 'pre_filter.pt']]

    def download(self):
        pass

    def process(self):
        
        all_graph_features = []
        data_list = []
        design_names = [f for f in os.listdir(self.root) if f not in ['processed', 'raw']]
        # print(f"Design names found: {design_names}")

        print("Collecting graph features for normalization...")
        for design_name in design_names:
            master = pd.read_csv(osp.join('dataset_prep', 'master.csv'), index_col=0)
            if design_name not in master.columns:
                print(f"Design {design_name} not in master, creating master file...")
                os.system(f"python dataset_prep/make_master_file.py --design_name {design_name}")

            dataset = PygNodePropPredDataset(name=f'{design_name}', root=self.root)
            data = dataset[0]

            data.y = data.y.T  
            data.graph_feature = data.graph_feature.T  
            data['desName'] = design_name

            all_graph_features.append(data.graph_feature.numpy())
            data_list.append(data)
            print(f"Collected graph_feature for {design_name}. Total count: {len(all_graph_features)}")

        all_graph_features = np.vstack(all_graph_features).astype(np.float32)

        scaler = StandardScaler()
        scaler.fit(all_graph_features)
        print("Computed StandardScaler for graph features.")

        joblib.dump(scaler, self.scaler_path)
        print(f"Saved scaler to {self.scaler_path}")
        print("Normalizing and saving processed data...")
        for idx, data in enumerate(data_list):
            normalized_graph_features = scaler.transform(data.graph_feature.numpy().reshape(1, -1)).squeeze()
            data.graph_feature = torch.tensor(normalized_graph_features, dtype=torch.float).reshape(1, -1)
            desName = data['desName']
            print(f"Processing {desName}")
            print("Original data:", dataset[0])
            print("Normalized data:", data)
            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
        print("Data processing and normalization completed.")

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

class BestMetricsTracker:
    def __init__(self):
        self.best_top1_metrics = (0.0, 0.0, 0.0)
        self.best_top1_model_state = None
        self.best_top1_params = None
        self.best_top3_metrics = (0.0, 0.0, 0.0)
        self.best_top3_model_state = None
        self.best_top3_params = None

    def update_best_top1(self, metrics, model_state, params):
        if metrics[0] > self.best_top1_metrics[0]:
            self.best_top1_metrics = metrics
            self.best_top1_model_state = model_state
            self.best_top1_params = params
            return True
        return False

    def update_best_top3(self, metrics, model_state, params):
        if metrics[2] > self.best_top3_metrics[2]:
            self.best_top3_metrics = metrics
            self.best_top3_model_state = model_state
            self.best_top3_params = params
            return True
        return False

def create_callback(dump_dir, tracker, top_n=3):
    def callback(study, trial):
        print(f"[Callback] Trial {trial.number} completed. Checking for new best metrics...")

        # # Handle new best Top-1 metrics
        # if tracker:
        #     best_top1_metrics = tracker.best_top1_metrics
        #     best_top1_model_state = tracker.best_top1_model_state
        #     best_top1_params = tracker.best_top1_params
            
        #     print(f"🚀 New Global Best Top-1 Accuracy: {best_top1_metrics[0]:.4f} (Trial {trial.number})")
        #     print(f"Corresponding Top-1, Top-2, Top-3 Accuracies: {best_top1_metrics}")
        #     print(f"Corresponding Best Hyperparameters: {best_top1_params}")
            
        #     # Save the best Top-1 model and related information
        #     model_path = os.path.join(dump_dir, 'best_top1_model.pt')
        #     torch.save(best_top1_model_state, model_path)
            
        #     with open(os.path.join(dump_dir, 'best_top1_metrics.pkl'), 'wb') as f:
        #         pickle.dump(best_top1_metrics, f)
            
        #     with open(os.path.join(dump_dir, 'best_top1_params.pkl'), 'wb') as f:
        #         pickle.dump(best_top1_params, f)
        
        # # Handle new best Top-3 metrics
        # if tracker:
        #     best_top3_metrics = tracker.best_top3_metrics
        #     best_top3_model_state = tracker.best_top3_model_state
        #     best_top3_params = tracker.best_top3_params
            
        #     print(f"🌟 New Global Best Top-3 Accuracy: {best_top3_metrics[2]:.4f} (Trial {trial.number})")
        #     print(f"Corresponding Top-1, Top-2, Top-3 Accuracies: {best_top3_metrics}")
        #     print(f"Corresponding Best Hyperparameters: {best_top3_params}")
            
        #     # Save the best Top-3 model and related information
        #     model_path = os.path.join(dump_dir, 'best_top3_model.pt')
        #     torch.save(best_top3_model_state, model_path)
            
        #     with open(os.path.join(dump_dir, 'best_top3_metrics.pkl'), 'wb') as f:
        #         pickle.dump(best_top3_metrics, f)
            
        #     with open(os.path.join(dump_dir, 'best_top3_params.pkl'), 'wb') as f:
        #         pickle.dump(best_top3_params, f)
        
        # Additional callback logic (e.g., saving top_n trials) can be added here
    
    return callback
def run_study(args, train_dl_msb, test_dl_msb, valid_dl_msb, DUMP_DIR):
    # Create an Optuna study
    study = optuna.create_study(
        direction='maximize',  # Change to 'maximize' if optimizing for accuracy
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.load_if_exists
    )
    
    tracker = BestMetricsTracker()
    
    # Create a partial function for the objective
    objective_with_args = partial(
        objective, 
        args=args, 
        train_dl_msb=train_dl_msb, 
        test_dl_msb=test_dl_msb,
        valid_dl_msb=valid_dl_msb,
        DUMP_DIR=DUMP_DIR,
        tracker=tracker  # Pass the tracker instance
    )
    
    # Define a callback
    callback = create_callback(DUMP_DIR, tracker, top_n=3)
    
    # Optimize the objective function
    study.optimize(objective_with_args, n_trials=args.n_trials, callbacks=[callback])
    
    # Save final best Top-1 Metrics and Params
    if tracker.best_top1_model_state is not None and tracker.best_top1_metrics is not None and tracker.best_top1_params is not None:
        with open(os.path.join(DUMP_DIR, 'final_best_top1_metrics.pkl'), 'wb') as f:
            pickle.dump(tracker.best_top1_metrics, f)
        with open(os.path.join(DUMP_DIR, 'final_best_top1_params.pkl'), 'wb') as f:
            pickle.dump(tracker.best_top1_params, f)
    
    # Save final best Top-3 Metrics and Params
    if tracker.best_top3_model_state is not None and tracker.best_top3_metrics is not None and tracker.best_top3_params is not None:
        with open(os.path.join(DUMP_DIR, 'final_best_top3_metrics.pkl'), 'wb') as f:
            pickle.dump(tracker.best_top3_metrics, f)
        with open(os.path.join(DUMP_DIR, 'final_best_top3_params.pkl'), 'wb') as f:
            pickle.dump(tracker.best_top3_params, f)
    
    return study
def process_msb_datasets(dataset_train_msb, dataset_test_msb, args):
    # Preprocess train dataset
    if isinstance(dataset_train_msb, list):
        dataset_train_msb = [preprocess(data, args) for data in dataset_train_msb]
    else:
        dataset_train_msb = preprocess(dataset_train_msb, args)

    # Convert to sparse tensor
    dataset_train_msb = [T.ToSparseTensor()(data) for data in dataset_train_msb]

    # Preprocess test dataset
    if isinstance(dataset_test_msb, list):
        dataset_test_msb = [preprocess(data, args) for data in dataset_test_msb]
    else:
        dataset_test_msb = preprocess(dataset_test_msb, args)

    # Convert to sparse tensor
    dataset_test_msb = [T.ToSparseTensor()(data) for data in dataset_test_msb]
    args.feature_size = dataset_train_msb[0].num_features
    # Split dataset into train and validation
    total_size_msb = len(dataset_train_msb)
    train_size_msb = (2 * total_size_msb) // 3
    val_size_msb = total_size_msb - train_size_msb

    train_dataset_msb = []
    val_dataset_msb = []
   
    for i in range(total_size_msb):
        if (i % 3) < 2 and len(train_dataset_msb) < train_size_msb:
            train_dataset_msb.append(dataset_train_msb[i])
        elif len(val_dataset_msb) < val_size_msb:
            val_dataset_msb.append(dataset_train_msb[i])
    print("hops",args.num_hops)
    print("data_dim",dataset_test_msb[0])
    
    # Load datasets into dataloaders
    train_dl_msb = DataLoader(train_dataset_msb, batch_size=args.batch_size, shuffle=False, num_workers=10)
    test_dl_msb = DataLoader(dataset_test_msb, batch_size=args.batch_size, shuffle=False, num_workers=10)
    valid_dl_msb = DataLoader(val_dataset_msb, batch_size=args.batch_size, shuffle=False, num_workers=10)
    
    return train_dl_msb, test_dl_msb, valid_dl_msb


#     return best_val_loss
def objective(trial, args, train_dl_msb, test_dl_msb, valid_dl_msb, DUMP_DIR, tracker=None):
    global best_top1_metrics, best_top1_model_state, best_top3_metrics, best_top3_model_state
    trial_params = {
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-1),
        'num_layers': trial.suggest_int('num_layers', 2, 4),
        'hidden_channels': trial.suggest_int('hidden_channels', 64, 256, step=16),
        'dropout': trial.suggest_uniform('dropout', 0.1, 0.7),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-3),
        'heads': trial.suggest_categorical('heads', [1, 2, 4, 8, 16]),
        'attn_dropout': trial.suggest_uniform('attn_dropout', 0.0, 0.5),
        'use_bias': trial.suggest_categorical('use_bias', [True, False]),
    }
    
    for key, value in trial_params.items():
        setattr(args, key, value)
    
    heads = trial_params['heads']
    num_hops = args.num_hops
    
    if num_hops <= 2:
        max_batch_size = 128
    elif num_hops <= 4:
        max_batch_size = 64
    else:
        max_batch_size = 32
    
    max_multiplier = max_batch_size // heads
    
    if max_multiplier < 1:
        return float('inf')
    
    batch_size_multiplier = trial.suggest_int('batch_size_multiplier', 1, max_multiplier)
    batch_size = heads * batch_size_multiplier
    args.batch_size = batch_size
    
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    torch.manual_seed(42)
    np.random.seed(42)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    
    model_msb = SimpleGNN1_HOGA(args).to(device)
    optimizer_msb = torch.optim.Adam(
        model_msb.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    scaler = GradScaler()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_msb, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    best_val_score = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        # Training
        train_loss_msb_value = train(model_msb, device, train_dl_msb, optimizer_msb, task=2)
    
        torch.cuda.empty_cache()
    
        # Validation
        valid_loss_msb_value, accuracy, top2_accuracy, top3_accuracy = evaluate1(model_msb, device, valid_dl_msb, task=2)
        test_loss_msb_value, accuracy1, top2_accuracy1, top3_accuracy1 = evaluate1(model_msb, device, test_dl_msb, task=2)
        best_test_metrics = (accuracy1, top2_accuracy1, top3_accuracy1)
        print(f"Accuracy (Top-1): {accuracy1}")
        print(f"Accuracy (Top-2): {top2_accuracy1}")
        print(f"Accuracy (Top-3): {top3_accuracy1}")
        # Report intermediate objective value
        trial.report(valid_loss_msb_value, epoch)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()    
        
        if tracker.update_best_top1(best_test_metrics, model_msb.state_dict(), trial_params):
            print(f"🚀 New Global Best Top-1 Accuracy: {tracker.best_top1_metrics[0]:.4f} (Trial {trial.number})")
            print(f"Corresponding Top-1, Top-2, Top-3 Accuracies: {tracker.best_top1_metrics}")
            print(f"Corresponding Best Hyperparameters: {tracker.best_top1_params}")
            
            model_path = os.path.join(DUMP_DIR, 'best_top1_model.pt')
            torch.save(tracker.best_top1_model_state, model_path)
            
            with open(os.path.join(DUMP_DIR, 'best_top1_metrics.pkl'), 'wb') as f:
                pickle.dump(tracker.best_top1_metrics, f)
            
            with open(os.path.join(DUMP_DIR, 'best_top1_params.pkl'), 'wb') as f:
                pickle.dump(tracker.best_top1_params, f)
        
        if tracker.update_best_top3(best_test_metrics, model_msb.state_dict(), trial_params):
            print(f"🌟 New Global Best Top-3 Accuracy: {tracker.best_top3_metrics[2]:.4f} (Trial {trial.number})")
            print(f"Corresponding Top-1, Top-2, Top-3 Accuracies: {tracker.best_top3_metrics}")
            print(f"Corresponding Best Hyperparameters: {tracker.best_top3_params}")
            
            model_path = os.path.join(DUMP_DIR, 'best_top3_model.pt')
            torch.save(tracker.best_top3_model_state, model_path)
            
            with open(os.path.join(DUMP_DIR, 'best_top3_metrics.pkl'), 'wb') as f:
                pickle.dump(tracker.best_top3_metrics, f)
            
            with open(os.path.join(DUMP_DIR, 'best_top3_params.pkl'), 'wb') as f:
                pickle.dump(tracker.best_top3_params, f)
        
    
    return best_val_score

def main():
    parser = argparse.ArgumentParser(description='mult16')
    parser.add_argument('--lmsb', type=str, choices=['MSB', 'LSB','LSB1'], default='MSB',
                        help='Specify the type: "MSB" or "LSB" (default: "MSB")')
    parser.add_argument('--op', type=str, default='none',
                        help='Specify the operation type (default: "none")')
    parser.add_argument('--task', type=int, required=True,
                        help='Specify the task number (required)')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True,
                        help='Specify the mode: "train" or "test" (required)')
    parser.add_argument('--root_path_train_lsb', type=str, default='dataset_ml_train_default_LSB',
                        help='Path to the training dataset for LSB (default: "dataset_ml_train_default_LSB")')
    parser.add_argument('--root_path_test_lsb', type=str, default='dataset_ml_test_default_LSB',
                        help='Path to the test dataset for LSB (default: "dataset_ml_test_default_LSB")')
    parser.add_argument('--root_path_train_lsb1', type=str, default='dataset_ml_train_default_LSB1',
                        help='Path to the training dataset for LSB (default: "dataset_ml_train_default_LSB")')
    parser.add_argument('--root_path_test_lsb1', type=str, default='dataset_ml_test_default_LSB1',
                        help='Path to the test dataset for LSB (default: "dataset_ml_test_default_LSB")')
    parser.add_argument('--root_path_train_msb', type=str, default='dataset_ml_train_default_MSB',
                        help='Path to the training dataset for MSB (default: "dataset_ml_train_default_MSB")')
    parser.add_argument('--root_path_test_msb', type=str, default='dataset_ml_test_default_MSB',
                        help='Path to the test dataset for MSB (default: "dataset_ml_test_default_MSB")')
    parser.add_argument('--root_path_train_lsb_hoga', type=str, default='dataset_ml_train_default_LSB',
                     help='Path to the training dataset for LSB (default: "dataset_ml_train_default_LSB")')
    parser.add_argument('--root_path_test_lsb_hoga', type=str, default='dataset_ml_test_default_LSB',
                     help='Path to the test dataset for LSB (default: "dataset_ml_test_default_LSB")')

    parser.add_argument('--root_path_train_msb_hoga', type=str, default='dataset_ml_train_default_MSB',
                        help='Path to the training dataset for MSB (default: "dataset_ml_train_default_MSB")')
    parser.add_argument('--root_path_test_msb_hoga', type=str, default='dataset_ml_test_default_MSB',
                        help='Path to the test dataset for MSB (default: "dataset_ml_test_default_MSB")')

    parser.add_argument('--bits', type=int, default=8)
    parser.add_argument('--bits_test', type=int, default=64)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--attn_dropout', type=int, default=0)
    parser.add_argument('--use_bias', type=bool, default=False, help='Whether to use bias in linear layers')
    parser.add_argument('--attn_type', type=str, default="vanilla")
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--hoga', type=int, default=0)
    # parser.add_argument('--op', type=str, default="default_U")
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--lr', type=float, default= 5e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--num_hops', type=int, default=2)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--mapped', type=int, default=0)
    parser.add_argument('--lda1', type=int, default=5)
    parser.add_argument('--lda2', type=int, default=1)
    parser.add_argument('--root_dir', type=str, default='data_ml')
    # parser.add_argument('--directed', action='True')
    parser.add_argument('--test_all_bits', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--pretrain',type=int,default=0)
    parser.add_argument('--num_fc_layer', type=int, default=1)
    parser.add_argument('--num_n_classes_task0', type=int, default=2)
    parser.add_argument('--num_n_classes_task1', type=int, default=5)
    parser.add_argument('--num_n_classes_task2', type=int, default=2)
    parser.add_argument('--num_n_classes_task3', type=int, default=9)
    parser.add_argument('--gnn_embedding_dim', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--study_name', type=str, default='optuna_study', help='Name of the Optuna study')
    parser.add_argument('--storage', type=str, default=None, help='Optuna storage URL for RDB backend')
    parser.add_argument('--load_if_exists', action='store_true', help='Load the study if it exists')
    parser.add_argument('--pooling_ratio', type=int, default=0.2)
   
   
   
    args = parser.parse_args()
    args.graph_feature_size =4
    # args.directed=True
    args.directed=False
    print(args)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    # device = torch.device('cpu') ## cpu for now only

    if not os.path.exists(f'models/'):
        os.makedirs(f'models/')

    if args.mapped == 1:
        suffix ="_7nm_mapped"
    elif args.mapped == 2:
        suffix ="_mapped"
    else:
        suffix = ''
    root_path_train_lsb = args.root_path_train_lsb
    root_path_test_lsb = args.root_path_test_lsb
    # root_path_train_lsb1 = args.root_path_train_lsb1
    # root_path_test_lsb1 = args.root_path_test_lsb1
    root_path_train_msb = args.root_path_train_msb
    root_path_test_msb = args.root_path_test_msb
    
    DUMP_DIR = "dump"
    if not os.path.exists(DUMP_DIR):
        os.makedirs(DUMP_DIR)
    current_dir = os.getcwd()
    # master1 = pd.read_csv(osp.join('dataset_prep', 'master1.csv'), index_col=0)
    
    # master1.to_csv(osp.join('dataset_prep', 'master.csv'))
        ########################loading LSB training data################
    # print(f"Loading LSB dataset from {root_path_train_lsb}")
    # #print("args.num_hops:", args.num_hops)
    # processed_train_lsb = os.path.join(root_path_train_lsb, 'processed')
    # if not os.path.exists(processed_train_lsb):
    #     os.makedirs(processed_train_lsb, exist_ok=True)    
    # dataset_lsb = MyOwnDataset(root=root_path_train_lsb, args=args)
    
    # print("data", dataset_lsb[0])
    
    # processed_dir_train_lsb = os.path.join(current_dir, 'processed_data_4_train_lsb_2')
    # # processed_dir_train_lsb = os.path.join(current_dir, 'processed_data_4_train_lsb')
    # if not os.path.exists(processed_dir_train_lsb):
    #     os.makedirs(processed_dir_train_lsb, exist_ok=True)
    # torch.save(dataset_lsb, os.path.join(processed_dir_train_lsb, 'data_processed_lsb.pt'))
    # dataset_train_lsb = torch.load(os.path.join(processed_dir_train_lsb, 'data_processed_lsb.pt'))
    # print(f"Number of samples in the LSB training dataset: {len(dataset_train_lsb)}")
    # print("data", dataset_train_lsb[0])

    # ########################loading LSB testing data################
    # print(f"Loading LSB dataset from {root_path_test_lsb}")
    # processed_test_lsb = os.path.join(root_path_test_lsb, 'processed')
    # if not os.path.exists(processed_test_lsb):
    #     os.makedirs(processed_test_lsb, exist_ok=True) 
    # dataset_test_lsb = MyOwnDataset(root=root_path_test_lsb, args=args)
    # processed_dir_test_lsb = os.path.join(current_dir, 'processed_data_4_test_lsb')
    # if not os.path.exists(processed_dir_test_lsb):
    #     os.makedirs(processed_dir_test_lsb, exist_ok=True)
    # torch.save(dataset_test_lsb, os.path.join(processed_dir_test_lsb, 'data_processed_lsb.pt'))

    # #########################splitting LSB dataset################
    # total_size_lsb = len(dataset_train_lsb)
        
    # train_dataset_lsb = []
    # val_dataset_lsb = []
    
    # for i in range(total_size_lsb):
    #     if (i % 3) < 2 and len(train_dataset_lsb) < train_size_lsb:
    #     elif len(val_dataset_lsb) < val_size_lsb:
            

    # ######################### load LSB dataset into dataloader###############
    # train_dl_lsb = DataLoader(train_dataset_lsb, batch_size=args.batch_size, shuffle=False, num_workers=10)
    # test_dl_lsb = DataLoader(dataset_test_lsb, batch_size=args.batch_size, shuffle=False, num_workers=10)
    # valid_dl_lsb = DataLoader(val_dataset_lsb, batch_size=args.batch_size, shuffle=False, num_workers=10)
    # print(f"LSB Training dataset size: {len(train_dataset_lsb)}")
    # print(f"LSB Validation dataset size: {len(val_dataset_lsb)}")
    # print(f"LSB Testing dataset size: {len(dataset_test_lsb)}")

 

    # if args.mode == 'train':
    #     for ep in range(1, args.num_epochs + 1):
    #         print(f"\nEpoch [{ep}/{args.num_epochs}]")
    #         # LSB Training
    #         print("\nTraining LSB model..")
    #         train_loss_lsb_value = train(model_lsb, device, train_dl_lsb, optimizer_lsb, task=1)     
    #         # Evaluation for LSB
    #         print("\nEvaluating LSB model..")
    
    #         # Save best models for LSB
    #         if validLoss_lsb_value < validLossOpt_lsb:
    #             validLossOpt_lsb = validLoss_lsb_value
    #             bestValEpoch_lsb = ep
    #             torch.save(model_lsb.state_dict(), osp.join(DUMP_DIR, f'model_lsb_{args.op}.pt'))
    
    #         print(f"Best LSB Val epoch: {bestValEpoch_lsb} with loss: {validLossOpt_lsb:.3f}")
    #         print({'LSB Train loss': train_loss_lsb_value, 'LSB Validation loss': validLoss_lsb_value})
    
    #         valid_curve_lsb.append(validLoss_lsb_value)
    #         train_loss_lsb.append(train_loss_lsb_value)
    
    #     with open(osp.join(DUMP_DIR, f'valid_curve_lsb_{args.op}.pkl'), 'wb') as f:
    #         pickle.dump(valid_curve_lsb, f)
    #     with open(osp.join(DUMP_DIR, f'train_loss_lsb_{args.op}.pkl'), 'wb') as f:
    #         pickle.dump(train_loss_lsb, f)
    
    #     plotChart([i + 1 for i in range(len(valid_curve_lsb))], valid_curve_lsb, "# Epochs", "Loss", "Validation loss LSB", "Validation Loss LSB", DUMP_DIR)
    #     plotChart([i + 1 for i in range(len(train_loss_lsb))], train_loss_lsb, "# Epochs", "Loss", "Training loss LSB", "Training Loss LSB", DUMP_DIR)
    
    # elif args.mode == 'test':
    #     model_lsb.load_state_dict(torch.load(osp.join(DUMP_DIR, f'model_lsb_{args.op}.pt')))
    #     print("Loaded model_lsb.pt")
            
    #################################################
    
    print(f"Loading MSB dataset from {root_path_train_msb}")
    processed_train_msb = os.path.join(root_path_train_msb, 'processed')
    if not os.path.exists(processed_train_msb):
        os.makedirs(processed_train_msb, exist_ok=True)    
    if args.hoga==0:
       dataset_msb = MyOwnDataset(root=root_path_train_msb, args=args)
       print("data", dataset_msb[0])
       processed_dir_train_msb = os.path.join(current_dir, 'processed_data_4_train_msb2')
       if not os.path.exists(processed_dir_train_msb):
              os.makedirs(processed_dir_train_msb, exist_ok=True)
       torch.save(dataset_msb, os.path.join(processed_dir_train_msb, 'data_processed_msb.pt'))
    else:
       dataset_msb = MyOwnDataset_hoga(root=root_path_train_msb, args=args)
       print("data", dataset_msb[0])
       processed_dir_train_msb = os.path.join(current_dir, 'processed_data_4_train_msb2')
       if not os.path.exists(processed_dir_train_msb):
              os.makedirs(processed_dir_train_msb, exist_ok=True)
       torch.save(dataset_msb, os.path.join(processed_dir_train_msb, 'data_processed_msb.pt'))

    dataset_train_msb = torch.load(os.path.join(processed_dir_train_msb, 'data_processed_msb.pt'))
    updated_dataset = []

    

    #model_lsb = SimpleGNN1_HOGA(args).to(device)
    #model_msb = SimpleGNN1_HOGA(args).to(device)
    # if args.pretrain == 1:
    #     model_lsb.load_state_dict(torch.load(osp.join(DUMP_DIR, 'model_lsb.pt')))
    #     model_msb.load_state_dict(torch.load(osp.join(DUMP_DIR, 'model_msb.pt')))
    learning_rate = args.lr
    valid_curve_lsb = []
    train_loss_lsb = []
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)
    validLossOpt_lsb = float('inf')  # Start with infinity for LSB valid loss
    bestValEpoch_lsb = 0
    validLossOpt_msb = float('inf') 
    for data in tqdm(dataset_train_msb, desc="Updating MSB Dataset", file=sys.stdout):
        desName = data.desName
        
        edge_index = data.edge_index
        
        graph_features = extract_graph_features(edge_index)
        graph_features = graph_features.unsqueeze(0)
        
        updated_graph_feature = torch.cat((data.graph_feature, graph_features), dim=1)
        data.graph_feature = updated_graph_feature
        
        updated_dataset.append(data)

    torch.save(updated_dataset, os.path.join(processed_dir_train_msb, 'data_processed_msb_updated_train.pt'))
   
    print(dataset_train_msb[0])
    dataset_train_msb = torch.load(os.path.join(processed_dir_train_msb, 'data_processed_msb_updated_train.pt'))
    print(f"Number of samples in the MSB training dataset: {len(dataset_train_msb)}")
   
   
   
   
   
   
   

   
    # dataset_train_msb = preprocess(dataset_train_msb ,args)
    # dataset_train_msb = T.ToSparseTensor()(dataset_train_msb )
    print(dataset_train_msb[0])
    
    ########################loading MSB testing data################
    print(f"Loading MSB dataset from {root_path_test_msb}")
    if args.hoga == 0:
        processed_test_msb = os.path.join(root_path_test_msb, 'processed')
        if not os.path.exists(processed_test_msb):
            os.makedirs(processed_test_msb, exist_ok=True)    
        dataset_test_msb = MyOwnDataset(root=root_path_test_msb, args=args)
    
        processed_dir_test_msb = os.path.join(current_dir, 'processed_data_4_test_msb')
        
        if not os.path.exists(processed_dir_test_msb):
            os.makedirs(processed_dir_test_msb, exist_ok=True)
        torch.save(dataset_test_msb, os.path.join(processed_dir_test_msb, 'data_processed_msb.pt'))
    # if switch == 1:
    #    dataset_test_msb = torch.load(os.path.join(processed_dir_test_msb, 'data_processed_msb_updated_test.pt'))
    #    print(f"Loaded updated dataset from {os.path.join(processed_dir_test_msb, 'data_processed_msb_updated_test.pt')}")
    # else:

    updated_test_dataset = []
    for data in tqdm(dataset_test_msb, desc="Updating MSB Test Dataset", file=sys.stdout):
            desName = data.desName
            edge_index = data.edge_index
            graph_features = extract_graph_features(edge_index)
            graph_features = graph_features.unsqueeze(0)
            updated_graph_feature = torch.cat((data.graph_feature,graph_features), dim=1)  
            data.graph_feature = updated_graph_feature
            updated_test_dataset.append(data)
    
    processed_dir_test_msb = os.path.join(current_dir, 'processed_data_4_test_msb')
    if not os.path.exists(processed_dir_test_msb):
        os.makedirs(processed_dir_test_msb, exist_ok=True)
    torch.save(updated_test_dataset, os.path.join(processed_dir_test_msb, 'data_processed_msb_updated_test.pt'))
    


    dataset_test_msb = MyOwnDataset(root=root_path_test_msb, args=args)
    dataset_test_msb = torch.load(os.path.join(processed_dir_test_msb, 'data_processed_msb_updated_test.pt'))
    # if isinstance(dataset_train_msb, list):
    #     dataset_train_msb = [preprocess(data, args) for data in dataset_train_msb]
    # else:
    #     dataset_train_msb = preprocess(dataset_train_msb, args)
    
    # dataset_train_msb = [T.ToSparseTensor()(data) for data in dataset_train_msb]
    
    # if isinstance(dataset_test_msb, list):
    #     dataset_test_msb = [preprocess(data, args) for data in dataset_test_msb]
    # else:
    #     dataset_test_msb = preprocess(dataset_test_msb, args)
    # dataset_test_msb = [T.ToSparseTensor()(data) for data in dataset_test_msb]
    
    # total_size_msb = len(dataset_train_msb)
    
    # train_dataset_msb = []
    # val_dataset_msb = []
    
    # for i in range(total_size_msb):
    #     if (i % 3) < 2 and len(train_dataset_msb) < train_size_msb:
    #     elif len(val_dataset_msb) < val_size_msb:
    
    # ######################### load MSB dataset into dataloader###############
    # train_dl_msb = DataLoader(train_dataset_msb, batch_size=args.batch_size, shuffle=False, num_workers=10)
    # test_dl_msb = DataLoader(dataset_test_msb, batch_size=args.batch_size, shuffle=False, num_workers=10)
    # valid_dl_msb = DataLoader(val_dataset_msb, batch_size=args.batch_size, shuffle=False, num_workers=10)
 
 
    
    
    # train_dl_msb=[]
    # test_dl_msb=[]
    # valid_dl_msb=[]
    if args.mode == 'train':
        args.num_hops = 2
        train_dl_msb, test_dl_msb, valid_dl_msb = process_msb_datasets(dataset_train_msb, dataset_test_msb, args)
        study = run_study(args, train_dl_msb, test_dl_msb, valid_dl_msb, DUMP_DIR)
        # args.num_hops = 4
        # train_dl_msb, test_dl_msb, valid_dl_msb = process_msb_datasets(dataset_train_msb, dataset_test_msb, args)
       # study = run_study(args, train_dl_msb, test_dl_msb, valid_dl_msb, DUMP_DIR)
        # args.num_hops = 6
        # train_dl_msb, test_dl_msb, valid_dl_msb = process_msb_datasets(dataset_train_msb, dataset_test_msb, args)
        # study = run_study(args, train_dl_msb, test_dl_msb, valid_dl_msb, DUMP_DIR)
        # args.num_hops = 8
        # train_dl_msb, test_dl_msb, valid_dl_msb = process_msb_datasets(dataset_train_msb, dataset_test_msb, args)
        #study = run_study(args, train_dl_msb, test_dl_msb, valid_dl_msb, DUMP_DIR)
        # study = optuna.create_study(
        #     direction='minimize',  # Assuming you want to minimize validation loss
        #     study_name=args.study_name,
        #     storage=args.storage,
        #     load_if_exists=args.load_if_exists
        # )
        # objective_with_args = partial(
        #     objective, 
        #     args=args, 
        #     train_dl_msb=train_dl_msb, 
        #     test_dl_msb=test_dl_msb,
        #     valid_dl_msb=valid_dl_msb,
        #     DUMP_DIR=DUMP_DIR
        # )
        # callback = create_callback(DUMP_DIR, args)
        # # Optimize the objective function
        # study.optimize(objective_with_args, n_trials=args.n_trials, callbacks=[callback])
        
        # Print study statistics
        
    else :
        filename = f"best_trial_{args.op}_hops_{args.num_hops}.pkl"
        pickle.dump(trial, f)
        filename = f"best_trial_{args.op}.pkl"
        file_path = os.path.join(DUMP_DIR, filename)
        
        with open(file_path, 'rb') as f:
             best_trial = pickle.load(f)
        best_params = best_trial.params
        print("Best Parameters:")
        for key, value in best_params.items():
            print(f"{key}: {value}")
        args.num_layers = best_params.get('num_layers', args.num_layers)
        args.hidden_channels = best_params.get('hidden_channels', args.hidden_channels)
        args.dropout = best_params.get('dropout', args.dropout)
        args.weight_decay = best_params.get('weight_decay', args.weight_decay)
        args.batch_size = best_params.get('batch_size', args.batch_size)
        args.heads = best_params.get('heads', args.heads)
        args.num_hops = best_params.get('num_hops', args.num_hops)
        args.attn_dropout = best_params.get('attn_dropout', args.attn_dropout)
        args.attn_type = best_params.get('attn_type', args.attn_type)
        args.use_bias = best_params.get('use_bias', args.use_bias)
        args.gnn_embedding_dim = best_params.get('gnn_embedding_dim', args.gnn_embedding_dim)
        args.lr = best_params.get('lr', args.lr)
        
        model_msb = SimpleGNN1(args).to(device)
        
        model_msb.load_state_dict(torch.load(os.path.join(DUMP_DIR, f'model_msb_{args.op}.pt')), strict=False)
        print("Loaded model_msb.pt")
      
    # ######################### load MSB dataset into dataloader###############
    # train_dl_msb, test_dl_msb, valid_dl_msb = process_msb_datasets(dataset_train_msb, dataset_test_msb, args)
    # train_dl_msb = DataLoader(train_dataset_msb, batch_size=args.batch_size, shuffle=False, num_workers=10)
    # test_dl_msb = DataLoader(dataset_test_msb, batch_size=args.batch_size, shuffle=False, num_workers=10)
    # valid_dl_msb = DataLoader(val_dataset_msb, batch_size=args.batch_size, shuffle=False, num_workers=10)
    # print(f"MSB Training dataset size: {len(train_dataset_msb)}")
    # print(f"MSB Validation dataset size: {len(val_dataset_msb)}")
    # print(f"MSB Testing dataset size: {len(dataset_test_msb)}")
    # print(train_dataset_msb[1])   
    # print(val_dataset_msb[1])
    # print(dataset_test_msb[1])
    # bestValEpoch_msb = 0
    # validLossOpt_msb = float('inf') 
    # valid_curve_msb = []
    # train_loss_msb = []
    # if args.mode == 'train':
    #     for ep in range(1, args.num_epochs + 1):
    #         print(f"\nEpoch [{ep}/{args.num_epochs}]")
    #         print("\nTraining MSB model..")
    #         train_loss_msb_value = train(model_msb, device, train_dl_msb, optimizer_msb, task=2)
    #         print("\nEvaluating MSB model..")
    #         validLoss_msb_value = evaluate(model_msb, device, valid_dl_msb, task=2)
    
    #         if validLoss_msb_value < validLossOpt_msb:
    #             validLossOpt_msb = validLoss_msb_value
    #             bestValEpoch_msb = ep
    #             torch.save(model_msb.state_dict(), osp.join(DUMP_DIR, f'model_msb_{args.op}.pt'))
    
    #         print(f"Best MSB Val epoch: {bestValEpoch_msb} with loss: {validLossOpt_msb:.3f}")
    #         print({'MSB Train loss': train_loss_msb_value, 'MSB Validation loss': validLoss_msb_value})
    
    #         valid_curve_msb.append(validLoss_msb_value)
    #         train_loss_msb.append(train_loss_msb_value)
    #     with open(osp.join(DUMP_DIR, f'valid_curve_msb_{args.op}.pkl'), 'wb') as f:
    #         pickle.dump(valid_curve_msb, f)
    #     with open(osp.join(DUMP_DIR, f'train_loss_msb_{args.op}.pkl'), 'wb') as f:
    #         pickle.dump(train_loss_msb, f)
    #     plotChart([i + 1 for i in range(len(valid_curve_msb))], valid_curve_msb, "# Epochs", "Loss", "Validation loss MSB", "Validation Loss MSB", DUMP_DIR)
    #     plotChart([i + 1 for i in range(len(train_loss_msb))], train_loss_msb, "# Epochs", "Loss", "Training loss MSB", "Training Loss MSB", DUMP_DIR)
    
    # elif args.mode == 'test':
    #     model_msb.load_state_dict(torch.load(osp.join(DUMP_DIR, f'model_msb_{args.op}.pt')))
    #     print("Loaded model_msb.pt")


    # print("Evaluating training data for TASK2...")
    # trainMSE_lsb, trainBatchData_lsb, incorrect_distribution_lsb = evaluate_plot(model_lsb, device, train_dl_lsb, task=1, num_classes=args.num_n_classes_task1)
    
    # print("Evaluating validation data for TASK2...")
    # validMSE_lsb, validBatchData_lsb, incorrect_distribution_lsb = evaluate_plot(model_lsb, device, valid_dl_lsb, task=1, num_classes=args.num_n_classes_task1)
    
    # print("Evaluating testing data for TASK2...")
    # testMSE_lsb, testBatchData_lsb, incorrect_distribution_lsb = evaluate_plot(model_lsb, device, test_dl_lsb, task=1, num_classes=args.num_n_classes_task1)
    # metrics_path = os.path.join(DUMP_DIR, 'global_best_test_metrics.pkl')

    # # Load and print the metrics
    # with open(metrics_path, 'rb') as f:
    #     best_metrics = pickle.load(f)
    
    # print("Best Test Metrics:", best_metrics)
    # print("Evaluating training data for TASK3...")
    # trainMSE_msb, trainBatchData_msb, incorrect_distribution_msb = evaluate_plot(model_msb, device, train_dl_msb, task=2, num_classes=args.num_n_classes_task3)
    
    # print("Evaluating validation data for TASK3...")
    # validMSE_msb, validBatchData_msb, incorrect_distribution_msb = evaluate_plot(model_msb, device, valid_dl_msb, task=2, num_classes=args.num_n_classes_task3)
    
    # print("Evaluating testing data for TASK3...")
    # testMSE_msb, testBatchData_msb, incorrect_distribution_msb = evaluate_plot(model_msb, device, test_dl_msb, task=2, num_classes=args.num_n_classes_task3)
    # test_loss_msb_value, accuracy1, top2_accuracy1, top3_accuracy1 = evaluate(model_msb, device, test_dl_msb, task=2)
    # print("evaluate msb test acc",accuracy1,top2_accuracy1,top3_accuracy1)
    # filepath_model1 = osp.join(DUMP_DIR, f'pred_stage2_{args.op}.txt')
    # filepath_model2 = osp.join(DUMP_DIR, f'pred_stage3_{args.op}.txt')
    # filepath_model2_time = osp.join(DUMP_DIR, f'pred_stage3_{args.op}_time.txt')
    
    # #arc_inference_model1(model_lsb, device, test_dl_lsb, filepath_model1)
    # arc_inference_model2(model_msb, device, test_dl_msb, filepath_model2,filepath_model2_time)
    # print("********************")
    # print("Final run statistics")
    # print("********************")
    # if args.mode == 'train':
    #     # print("LSB Training loss per sample: {}".format(trainMSE_lsb))
    #     print("MSB Training loss per sample: {}".format(trainMSE_msb))
    # elif args.mode == 'test':
    #     # print("LSB Test loss per sample: {}".format(testMSE_lsb))
    #     print("MSB Test loss per sample: {}".format(testMSE_msb))
    # print("********************")

if __name__ == "__main__":
    main()