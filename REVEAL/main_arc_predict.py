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
from utils import *
from dataset_prep import PygNodePropPredDataset, Evaluator
from model import SynthNet,SimpleGNN1,SimpleGNN_SAGPool,BiDirectionalGraphSAGE,SimpleGraphTransformer
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

os.environ['TORCH_HOME'] = '/tmp/.cache/torch'
def add_edge_direction(dataset, directed=True):
    for data in dataset:
        num_edges = data.edge_index.size(1)
        
        if directed:
            edge_direction = torch.ones(num_edges, dtype=torch.long)
        else:
            src = data.edge_index[0]
            dst = data.edge_index[1]
            edge_direction = (dst > src).long()
        
        data.edge_direction = edge_direction
    
    return dataset
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

            # Ensure graph_feature has the expected dimensionality
            graph_feature = data.graph_feature.numpy().astype(np.float32)
            if graph_feature.ndim == 1:
                graph_feature = graph_feature.reshape(1, -1)
            if graph_feature.shape[0] > 1:
                graph_feature = graph_feature[:1]
            expected_dim = getattr(self.args, 'graph_feature_size', graph_feature.shape[1])
            current_dim = graph_feature.shape[1]
            if current_dim < expected_dim:
                print(f"Warning: graph_feature for {design_name} has {current_dim} columns; padding to {expected_dim}.")
                pad_width = expected_dim - current_dim
                graph_feature = np.pad(graph_feature, ((0, 0), (0, pad_width)), mode='constant')
            elif current_dim > expected_dim:
                print(f"Warning: graph_feature for {design_name} has {current_dim} columns; truncating to {expected_dim}.")
                graph_feature = graph_feature[:, :expected_dim]
            graph_feature = np.nan_to_num(graph_feature, nan=0.0, posinf=0.0, neginf=0.0)
            data.graph_feature = torch.from_numpy(graph_feature)

            all_graph_features.append(graph_feature)
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





def main():
    torch.cuda.set_device(1)
    torch.cuda.empty_cache()
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")


    if cuda_available:
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")

        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA not available. No GPU found or CUDA not properly installed.")
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
    parser.add_argument('--bits', type=int, default=8)
    parser.add_argument('--bits_test', type=int, default=64)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--attn_dropout', type=int, default=0)
    parser.add_argument('--use_bias', type=bool, default=False, help='Whether to use bias in linear layers')
    parser.add_argument('--attn_type', type=str, default="vanilla")
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--heads', type=int, default=4)
    # parser.add_argument('--op', type=str, default="default_U")
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--lr', type=float, default= 5e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_hops', type=int, default=7)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--mapped', type=int, default=0)
    parser.add_argument('--lda1', type=int, default=5)
    parser.add_argument('--lda2', type=int, default=1)
    parser.add_argument('--root_dir', type=str, default='data_ml')
    parser.add_argument('--directed', action='store_true')
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
    parser.add_argument('--feature_size', type=int, default=16)
    args = parser.parse_args()
    args.graph_feature_size =4
    #args.feature_size = 16 
    print(args)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    # device = torch.device('cpu') ## cpu for now only
    print(f"Allocated memory: {torch.cuda.memory_allocated(device)} bytes")
    print(f"Reserved memory: {torch.cuda.memory_reserved(device)} bytes")
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
    root_path_train_lsb1 = args.root_path_train_lsb1
    root_path_test_lsb1 = args.root_path_test_lsb1
    root_path_train_msb = args.root_path_train_msb
    root_path_test_msb = args.root_path_test_msb
    
    DUMP_DIR = "dump"
    if not os.path.exists(DUMP_DIR):
        os.makedirs(DUMP_DIR)
    current_dir = os.getcwd()
    model_lsb = SimpleGNN1(args).to(device)
    # model_msb = SimpleGNN1(args).to(device)
    # model_msb = BiDirectionalGraphSAGE(args).to(device)
    model_msb = BiDirectionalGraphSAGE(args).to(device)
    # if args.pretrain == 1:
    #     model_lsb.load_state_dict(torch.load(osp.join(DUMP_DIR, 'model_lsb.pt')))
    #     model_msb.load_state_dict(torch.load(osp.join(DUMP_DIR, 'model_msb.pt')))
    learning_rate = args.lr
    optimizer_lsb = torch.optim.Adam(model_lsb.parameters(), lr=learning_rate)
    optimizer_msb = torch.optim.Adam(model_msb.parameters(), lr=learning_rate)
    valid_curve_lsb = []
    train_loss_lsb = []
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)
    validLossOpt_lsb = float('inf')  # Start with infinity for LSB valid loss
    bestValEpoch_lsb = 0
    validLossOpt_msb = float('inf') 
    # master1 = pd.read_csv(osp.join('dataset_prep', 'master1.csv'), index_col=0)
    
    # master1.to_csv(osp.join('dataset_prep', 'master.csv'))
        ########################loading LSB training data################
    print(f"Loading LSB dataset from {root_path_train_lsb}")
    #print("args.num_hops:", args.num_hops)
    processed_train_lsb = os.path.join(root_path_train_lsb, 'processed')
    if not os.path.exists(processed_train_lsb):
        os.makedirs(processed_train_lsb, exist_ok=True)    
    dataset_lsb = MyOwnDataset(root=root_path_train_lsb, args=args)
    
    print("data", dataset_lsb[0])
    
    processed_dir_train_lsb = os.path.join(current_dir, 'processed_data_4_train_lsb_2')
    # processed_dir_train_lsb = os.path.join(current_dir, 'processed_data_4_train_lsb')
    if not os.path.exists(processed_dir_train_lsb):
        os.makedirs(processed_dir_train_lsb, exist_ok=True)
    torch.save(dataset_lsb, os.path.join(processed_dir_train_lsb, 'data_processed_lsb.pt'))
    dataset_train_lsb = torch.load(os.path.join(processed_dir_train_lsb, 'data_processed_lsb.pt'))
    print(f"Number of samples in the LSB training dataset: {len(dataset_train_lsb)}")
    print("data", dataset_train_lsb[0])

    ########################loading LSB testing data################
    print(f"Loading LSB dataset from {root_path_test_lsb}")
    processed_test_lsb = os.path.join(root_path_test_lsb, 'processed')
    if not os.path.exists(processed_test_lsb):
        os.makedirs(processed_test_lsb, exist_ok=True) 
    dataset_test_lsb = MyOwnDataset(root=root_path_test_lsb, args=args)
    processed_dir_test_lsb = os.path.join(current_dir, 'processed_data_4_test_lsb')
    if not os.path.exists(processed_dir_test_lsb):
        os.makedirs(processed_dir_test_lsb, exist_ok=True)
    torch.save(dataset_test_lsb, os.path.join(processed_dir_test_lsb, 'data_processed_lsb.pt'))

    #########################splitting LSB dataset################
    total_size_lsb = len(dataset_train_lsb)
    train_size_lsb = int(2 * total_size_lsb / 3)
    val_size_lsb = total_size_lsb - train_size_lsb
        
    train_dataset_lsb = []
    val_dataset_lsb = []
    
    for i in range(total_size_lsb):
        if (i % 3) < 2 and len(train_dataset_lsb) < train_size_lsb:
            train_dataset_lsb.append(dataset_train_lsb[i])
        elif len(val_dataset_lsb) < val_size_lsb:
            val_dataset_lsb.append(dataset_train_lsb[i])
            

    # ######################### load LSB dataset into dataloader###############
    train_dl_lsb = DataLoader(train_dataset_lsb, batch_size=args.batch_size, shuffle=False, num_workers=10)
    test_dl_lsb = DataLoader(dataset_test_lsb, batch_size=args.batch_size, shuffle=False, num_workers=10)
    valid_dl_lsb = DataLoader(val_dataset_lsb, batch_size=args.batch_size, shuffle=False, num_workers=10)
    print(f"LSB Training dataset size: {len(train_dataset_lsb)}")
    print(f"LSB Validation dataset size: {len(val_dataset_lsb)}")
    print(f"LSB Testing dataset size: {len(dataset_test_lsb)}")

 

    if args.mode == 'train':
        for ep in range(1, args.num_epochs + 1):
            print(f"\nEpoch [{ep}/{args.num_epochs}]")
            # LSB Training
            print("\nTraining LSB model..")
            train_loss_lsb_value = train(model_lsb, device, train_dl_lsb, optimizer_lsb, task=1)     
            # Evaluation for LSB
            print("\nEvaluating LSB model..")
            validLoss_lsb_value = evaluate(model_lsb, device, valid_dl_lsb, task=1)
    
            # Save best models for LSB
            if validLoss_lsb_value < validLossOpt_lsb:
                validLossOpt_lsb = validLoss_lsb_value
                bestValEpoch_lsb = ep
                torch.save(model_lsb.state_dict(), osp.join(DUMP_DIR, f'model_lsb_{args.op}.pt'))
    
            print(f"Best LSB Val epoch: {bestValEpoch_lsb} with loss: {validLossOpt_lsb:.3f}")
            print({'LSB Train loss': train_loss_lsb_value, 'LSB Validation loss': validLoss_lsb_value})
    
            valid_curve_lsb.append(validLoss_lsb_value)
            train_loss_lsb.append(train_loss_lsb_value)
    
        with open(osp.join(DUMP_DIR, f'valid_curve_lsb_{args.op}.pkl'), 'wb') as f:
            pickle.dump(valid_curve_lsb, f)
        with open(osp.join(DUMP_DIR, f'train_loss_lsb_{args.op}.pkl'), 'wb') as f:
            pickle.dump(train_loss_lsb, f)
    
        plotChart([i + 1 for i in range(len(valid_curve_lsb))], valid_curve_lsb, "# Epochs", "Loss", "Validation loss LSB", "Validation Loss LSB", DUMP_DIR)
        plotChart([i + 1 for i in range(len(train_loss_lsb))], train_loss_lsb, "# Epochs", "Loss", "Training loss LSB", "Training Loss LSB", DUMP_DIR)
    
    elif args.mode == 'test':
        model_lsb.load_state_dict(torch.load(osp.join(DUMP_DIR, f'model_lsb_{args.op}.pt')))
        print("Loaded model_lsb.pt")
            
    #################################################
    switch = 2 

    print(f"Loading MSB dataset from {root_path_train_msb}")
    processed_train_msb = os.path.join(root_path_train_msb, 'processed')
    if not os.path.exists(processed_train_msb):
        os.makedirs(processed_train_msb, exist_ok=True)    
    dataset_msb = MyOwnDataset(root=root_path_train_msb, args=args)
    print("data", dataset_msb[0])
    processed_dir_train_msb = os.path.join(current_dir, 'processed_data_4_train_msb2')
    if not os.path.exists(processed_dir_train_msb):
        os.makedirs(processed_dir_train_msb, exist_ok=True)
    torch.save(dataset_msb, os.path.join(processed_dir_train_msb, 'data_processed_msb.pt'))


    if switch == 1:
        dataset_train_msb = torch.load(os.path.join(processed_dir_train_msb, 'data_processed_msb_updated_train.pt'))
        print(f"Loaded updated dataset from {os.path.join(processed_dir_train_msb, 'data_processed_msb_updated_train.pt')}")
    else: 
        dataset_train_msb = torch.load(os.path.join(processed_dir_train_msb, 'data_processed_msb.pt'))
        updated_dataset = []
    
        for data in tqdm(dataset_train_msb, desc="Updating MSB Dataset", file=sys.stdout):
            desName = data.desName
            
            edge_index = data.edge_index
            
            graph_features = extract_graph_features(edge_index)
            graph_features = graph_features.unsqueeze(0)
            
            updated_graph_feature = torch.cat((data.graph_feature, graph_features), dim=1)
            data.graph_feature = updated_graph_feature
            
            updated_dataset.append(data)
    
        torch.save(updated_dataset, os.path.join(processed_dir_train_msb, 'data_processed_msb_updated_train.pt'))
  

    dataset_train_msb = torch.load(os.path.join(processed_dir_train_msb, 'data_processed_msb_updated_train.pt'))
    print(f"Number of samples in the MSB training dataset: {len(dataset_train_msb)}")
    args.feature_size = dataset_train_msb[0].num_features
    print("args.feature_size", args.feature_size)
    ########################loading MSB testing data################
    print(f"Loading MSB dataset from {root_path_test_msb}")
    processed_test_msb = os.path.join(root_path_test_msb, 'processed')
    if not os.path.exists(processed_test_msb):
        os.makedirs(processed_test_msb, exist_ok=True)    
    dataset_test_msb = MyOwnDataset(root=root_path_test_msb, args=args)

    processed_dir_test_msb = os.path.join(current_dir, 'processed_data_4_test_msb')
    
    if not os.path.exists(processed_dir_test_msb):
        os.makedirs(processed_dir_test_msb, exist_ok=True)
    torch.save(dataset_test_msb, os.path.join(processed_dir_test_msb, 'data_processed_msb.pt'))
    if switch == 1:
       dataset_test_msb = torch.load(os.path.join(processed_dir_test_msb, 'data_processed_msb_updated_test.pt'))
       print(f"Loaded updated dataset from {os.path.join(processed_dir_test_msb, 'data_processed_msb_updated_test.pt')}")
    else:

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
    


    #dataset_test_msb = MyOwnDataset(root=root_path_test_msb, args=args)
    dataset_test_msb = torch.load(os.path.join(processed_dir_test_msb, 'data_processed_msb_updated_test.pt'))
    dataset_train_msb = add_edge_direction(dataset_train_msb, directed=True)
    dataset_test_msb = add_edge_direction(dataset_test_msb, directed=True)
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

        
    ######################### load MSB dataset into dataloader###############
    train_dl_msb = DataLoader(train_dataset_msb, batch_size=args.batch_size, shuffle=False, num_workers=10)
    test_dl_msb = DataLoader(dataset_test_msb, batch_size=args.batch_size, shuffle=False, num_workers=10)
    valid_dl_msb = DataLoader(val_dataset_msb, batch_size=args.batch_size, shuffle=False, num_workers=10)
    print(f"MSB Training dataset size: {len(train_dataset_msb)}")
    print(f"MSB Validation dataset size: {len(val_dataset_msb)}")
    print(f"MSB Testing dataset size: {len(dataset_test_msb)}")
    print(train_dataset_msb[1])   
    print(val_dataset_msb[1])
    print(dataset_test_msb[1])
    bestValEpoch_msb = 0
    validLossOpt_msb = float('inf') 
    valid_curve_msb = []
    train_loss_msb = []
    if args.mode == 'train':
        for ep in range(1, args.num_epochs + 1):
            print(f"\nEpoch [{ep}/{args.num_epochs}]")
            print("\nTraining MSB model..")
            train_loss_msb_value = train(model_msb, device, train_dl_msb, optimizer_msb, task=2)
            print("\nEvaluating MSB model..")
            validLoss_msb_value = evaluate(model_msb, device, valid_dl_msb, task=2)
    
            if validLoss_msb_value < validLossOpt_msb:
                validLossOpt_msb = validLoss_msb_value
                bestValEpoch_msb = ep
                torch.save(model_msb.state_dict(), osp.join(DUMP_DIR, f'model_msb_{args.op}.pt'))
    
            print(f"Best MSB Val epoch: {bestValEpoch_msb} with loss: {validLossOpt_msb:.3f}")
            print({'MSB Train loss': train_loss_msb_value, 'MSB Validation loss': validLoss_msb_value})
    
            valid_curve_msb.append(validLoss_msb_value)
            train_loss_msb.append(train_loss_msb_value)
        with open(osp.join(DUMP_DIR, f'valid_curve_msb_{args.op}.pkl'), 'wb') as f:
            pickle.dump(valid_curve_msb, f)
        with open(osp.join(DUMP_DIR, f'train_loss_msb_{args.op}.pkl'), 'wb') as f:
            pickle.dump(train_loss_msb, f)
        plotChart([i + 1 for i in range(len(valid_curve_msb))], valid_curve_msb, "# Epochs", "Loss", "Validation loss MSB", "Validation Loss MSB", DUMP_DIR)
        plotChart([i + 1 for i in range(len(train_loss_msb))], train_loss_msb, "# Epochs", "Loss", "Training loss MSB", "Training Loss MSB", DUMP_DIR)
    
    elif args.mode == 'test':
        model_msb.load_state_dict(torch.load(osp.join(DUMP_DIR, f'model_msb_{args.op}.pt')))
        print("Loaded model_msb.pt")


    print("Evaluating training data for TASK2...")
    trainMSE_lsb, trainBatchData_lsb, incorrect_distribution_lsb = evaluate_plot(model_lsb, device, train_dl_lsb, task=1, num_classes=args.num_n_classes_task1)
    
    print("Evaluating validation data for TASK2...")
    validMSE_lsb, validBatchData_lsb, incorrect_distribution_lsb = evaluate_plot(model_lsb, device, valid_dl_lsb, task=1, num_classes=args.num_n_classes_task1)
    
    print("Evaluating testing data for TASK2...")
    testMSE_lsb, testBatchData_lsb, incorrect_distribution_lsb = evaluate_plot(model_lsb, device, test_dl_lsb, task=1, num_classes=args.num_n_classes_task1)
    
    print("Evaluating training data for TASK3...")
    trainMSE_msb, trainBatchData_msb, incorrect_distribution_msb = evaluate_plot(model_msb, device, train_dl_msb, task=2, num_classes=args.num_n_classes_task3)
    
    print("Evaluating validation data for TASK3...")
    validMSE_msb, validBatchData_msb, incorrect_distribution_msb = evaluate_plot(model_msb, device, valid_dl_msb, task=2, num_classes=args.num_n_classes_task3)
    
    print("Evaluating testing data for TASK3...")
    testMSE_msb, testBatchData_msb, incorrect_distribution_msb = evaluate_plot(model_msb, device, test_dl_msb, task=2, num_classes=args.num_n_classes_task3)
    
    filepath_model1 = osp.join(DUMP_DIR, f'pred_stage2_{args.op}.txt')
    filepath_model2 = osp.join(DUMP_DIR, f'pred_stage3_{args.op}.txt')
    filepath_model2_time = osp.join(DUMP_DIR, f'pred_stage3_{args.op}_time.txt')
    
    arc_inference_model1(model_lsb, device, test_dl_lsb, filepath_model1)
    arc_inference_model2(model_msb, device, test_dl_msb, filepath_model2,filepath_model2_time)
    print("********************")
    print("Final run statistics")
    print("********************")
    if args.mode == 'train':
        # print("LSB Training loss per sample: {}".format(trainMSE_lsb))
        print("MSB Training loss per sample: {}".format(trainMSE_msb))
    elif args.mode == 'test':
        # print("LSB Test loss per sample: {}".format(testMSE_lsb))
        print("MSB Test loss per sample: {}".format(testMSE_msb))
    print("********************")

if __name__ == "__main__":
    main()