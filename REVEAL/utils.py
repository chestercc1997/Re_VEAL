import torch
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor
import torch.nn.functional as F
import numpy as np
from scipy.sparse import identity, diags
import argparse
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
from statistics import mean
from webbrowser import get
import matplotlib.pyplot as plt
import os.path as osp
import pandas as pd
from tqdm import tqdm
import sys
import os as os 
import networkx as nx
import time as time
from torch_geometric.utils import from_scipy_sparse_matrix

def graph2adj(adj):
    #hat_adj = adj + identity(adj.shape[0])
    hat_adj = adj
    degree_vec = hat_adj.sum(axis=1)
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.squeeze(np.asarray(np.power(degree_vec, -0.5)))
    d_inv_sqrt[np.isinf(d_inv_sqrt)|np.isnan(d_inv_sqrt)] = 0
    degree_matrix  = diags(d_inv_sqrt, 0)
    DAD = degree_matrix @ (hat_adj @ degree_matrix)
    AD = hat_adj @ (degree_matrix @ degree_matrix)
    DA = degree_matrix @ (degree_matrix @ hat_adj)

    return DAD, AD, DA
def adj_to_edge_index(adj_t):
    # Convert the SparseTensor to a scipy sparse matrix
    adj_matrix = adj_t.to_scipy(layout='csr')
    
    # Use from_scipy_sparse_matrix to get edge_index and edge_attr (weights)
    edge_index, edge_attr = from_scipy_sparse_matrix(adj_matrix)
    
    return edge_index
def preprocess(data, args):
    print("Preprocessing node features!!!!!!")
    nnodes = data.x.shape[0]
    if not args.directed:
        data.edge_index = to_undirected(data.edge_index, nnodes)
        row, col = data.edge_index
        adj = SparseTensor(row=row, col=col, sparse_sizes=(nnodes, nnodes))
        adj = adj.to_scipy(layout='csr')
        DAD, AD, DA = graph2adj(adj)
        norm_adj = SparseTensor.from_scipy(DAD).float()
        feat_lst = []
        feat_lst.append(data.x)
        high_order_features = data.x.clone()
        for _ in range(args.num_hops):
            high_order_features = norm_adj @ high_order_features
            #data.x = torch.cat((data.x, high_order_features), dim=1)
            feat_lst.append(high_order_features)
        data.x = torch.stack(feat_lst, dim=1)
        #data.num_features *= (1+args.num_hops)
    else:
        row, col = data.edge_index
        adj = SparseTensor(row=row, col=col, sparse_sizes=(nnodes, nnodes))
        adj = adj.to_scipy(layout='csr')
        _, _, DA = graph2adj(adj)
        _, _, DA_tran = graph2adj(adj.transpose())
        norm_adj = SparseTensor.from_scipy(DA).float()
        norm_adj_tran = SparseTensor.from_scipy(DA_tran).float()
        feat_lst = []
        feat_lst.append(data.x)
        high_order_features = data.x.clone()
        high_order_features_tran = data.x.clone()
        for _ in range(args.num_hops):
            high_order_features = norm_adj @ high_order_features
            high_order_features_tran = norm_adj_tran @ high_order_features_tran
            #data.x = torch.cat((data.x, high_order_features, high_order_features_tran), dim=1)
            feat_lst.append(high_order_features)
            feat_lst.append(high_order_features_tran)
        data.x = torch.stack(feat_lst, dim=1)
        #data.num_features *= (1+2*args.num_hops)

    return data

def all_numpy(obj):
    # Ensure everything is in numpy or int or float (no torch tensor)

    if isinstance(obj, dict):
        for key in obj.keys():
            all_numpy(obj[key])
    elif isinstance(obj, list):
        for i in range(len(obj)):
            all_numpy(obj[i])
    else:
        if not isinstance(obj, (np.ndarray, int, float)):
            return False

    return True






class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mapAttributesToTensor(data,areaDict,delayDict):
    area = data.area
    delay = data.delay
    minMaxArea = areaDict[data.desName[0]]
    minMaxDelay = delayDict[data.desName[0]]
    data.area = (area - minMaxArea[1])/(minMaxArea[0] - minMaxArea[1])
    data.delay = (delay - minMaxDelay[1]) / (minMaxDelay[0] - minMaxDelay[1])
    return data


def mse(y_pred,y_true):
    return mean_squared_error(y_true.view(-1,1).detach().cpu().numpy(),y_pred.view(-1,1).detach().cpu().numpy())

def mae(y_pred,y_true):
    return mean_absolute_error(y_true.view(-1,1).detach().cpu().numpy(),y_pred.view(-1,1).detach().cpu().numpy())

def doScatterPlot(batchLen,batchSize,batchData,dumpDir,trainMode):
    predList = []
    actualList = []
    designList = []
    for i in range(batchLen):
        numElemsInBatch = len(batchData[i][0])
        for batchID in range(numElemsInBatch):
            predList.append(batchData[i][0][batchID][0])
            actualList.append(batchData[i][1][batchID][0])
            designList.append(batchData[i][2][batchID][0])

    scatterPlotDF = pd.DataFrame({'designs': designList,
                                  'prediction': predList,
                                  'actual': actualList})

    uniqueDesignList = scatterPlotDF.designs.unique()

    for d in uniqueDesignList:
        designDF = scatterPlotDF[scatterPlotDF.designs == d]
        designDF.plot.scatter(x='actual', y='prediction', c='DarkBlue')
        plt.title(d)
        fileName = osp.join(dumpDir,"scatterPlot_"+trainMode+"_"+d+".png")
        #else:
        #    fileName = osp.join(dumpDir,"scatterPlot_test_"+d+".png")
        plt.savefig(fileName,fmt='png',bbox_inches='tight')


def getTopKSimilarityPercentage(list1,list2,topkpercent):
    listLen = len(list1)
    topKIndexSimilarity = int(topkpercent*listLen)
    Set1 = set(list1[:topKIndexSimilarity])
    Set2 = set(list2[:topKIndexSimilarity])
    numSimilarScripts = len(Set1.intersection(Set2))
    if topKIndexSimilarity >0:
        return (numSimilarScripts/topKIndexSimilarity)
    else:
        return 0


def doScatterAndTopKRanking(batchLen,batchSize,batchData,dumpDir,trainMode):
    predList = []
    actualList = []
    designList = []
    synthesisID = []

    for i in range(batchLen):
        numElemsInBatch = len(batchData[i][0])
        print("Batch: "+str(i),numElemsInBatch)
        for batchID in range(numElemsInBatch):
            predList.append(batchData[i][0][batchID][0])
            print(batchData[i][0][batchID][0])
            actualList.append(batchData[i][1][batchID][0])
         #   print(batchData[i][1][batchID][0])
            designList.append(batchData[i][2][batchID])
         #   print(batchData[i][2][batchID])
          #  synthesisID.append(int(batchData[i][3][batchID]))
          #  print(batchData[i][3][batchID])
    scatterPlotDF = pd.DataFrame({'designs': designList,
                                  #'synID': synthesisID,
                                  'prediction': predList,
                                  'actual': actualList})

    uniqueDesignList = scatterPlotDF.designs.unique()

    accuracyFile = osp.join(dumpDir, "topKaccuracy_" + trainMode + ".csv")
    accuracyFileWriter = open(accuracyFile,'w+')
    accuracyFileWriter.write("design,top1,top5,top10,top15,top20,top25"+"\n")
    endDelim = "\n"
    commaDelim = ","

    print("\nDataset type: "+trainMode)
    for d in uniqueDesignList:
        designDF = scatterPlotDF[scatterPlotDF.designs == d]
        designDF.plot.scatter(x='actual', y='prediction', c='DarkBlue')
        plt.title(d,weight='bold',fontsize=25)
        plt.xlabel('Actual', weight='bold', fontsize=25)
        plt.ylabel('Predicted', weight='bold', fontsize=25)
        fileName = osp.join(dumpDir,"scatterPlot_"+trainMode+"_"+d+".png")
        plt.savefig(fileName, format='png', bbox_inches='tight')
        desDF1 = designDF.sort_values(by=['actual'])
        desDF2 = designDF.sort_values(by=['prediction'])
        desDF1_synID = desDF1.synID.to_list()
        desDF2_synID = desDF2.synID.to_list()
        kPercentSimilarity = [0.01,0.05,0.1,0.15,0.2,0.25]
        accuracyFileWriter.write(d)
        for kPer in kPercentSimilarity:
            topKPercentSimilarity = getTopKSimilarityPercentage(desDF1_synID,desDF2_synID,kPer)
            accuracyFileWriter.write(commaDelim+str(topKPercentSimilarity))
        accuracyFileWriter.write(endDelim)
        desDF1.to_csv(osp.join(dumpDir,"desDF1_"+trainMode+"_"+d+".csv"),index=False)
        desDF2.to_csv(osp.join(dumpDir,"desDF2_"+trainMode+"_"+d+".csv"),index=False)
        mapeScore = mean_absolute_percentage_error(designDF.prediction.to_list(),designDF.actual.to_list())
        print("MAPE ("+d+"): "+str(mapeScore))
    accuracyFileWriter.close()




criterion = torch.nn.MSELoss()
def one_hot(labels, num_classes):
    return torch.eye(num_classes)[labels]

def evaluate_plot(model, device, dataloader, task, num_classes):
    model.eval()
    totalMSE = AverageMeter()
    batchData = []
    incorrect_counts = {}
    
    correct_top1 = 0
    correct_top2 = 0
    total = 0
    total_correct_num0=0
    total_correct_num1=0
    total_correct_num2=0
    total_correct_num3=0
    total_sample_num0=0
    total_sample_num1=0
    total_sample_num2=0
    total_sample_num3=0

    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader, desc="Iteration", file=sys.stdout)):
            batch = batch.to(device)
            pred_task1, pred_task2, pred_task3 = model(batch, task)
            lbl = batch.y.float()
            

            if task == 1:
                lbl_task = lbl[:, 1].long() - 1
                pred_scores = pred_task1
                pred_top1 = pred_scores.argmax(dim=-1).detach().cpu().numpy()
                accuracy = calculate_accuracy(pred_top1, lbl_task.view(-1).detach().cpu().numpy())
                correcut_num,sample_num=calculate_accuracy_total(pred_top1, lbl_task.view(-1).detach().cpu().numpy())
                total_correct_num0 +=correcut_num
                total_sample_num0 +=sample_num
                print("total_correct_num,total_sample_num",total_correct_num0,total_sample_num0)
                print(f"Accuracy (Top-1): {accuracy}")
            elif task == 2:
                lbl_task = lbl[:, 2].long() - 1
                pred_scores = pred_task3
                topk = [1, 2, 3]
                pred_topk = pred_scores.topk(max(topk), dim=-1)[1].detach().cpu().numpy()
                accuracy = calculate_accuracy(pred_topk[:, 0], lbl_task.view(-1).detach().cpu().numpy())
                top2_accuracy = (pred_topk[:, :2] == lbl_task.view(-1).detach().cpu().numpy()[:, None]).any(axis=1).mean()
                top3_accuracy = (pred_topk[:, :3] == lbl_task.view(-1).detach().cpu().numpy()[:, None]).any(axis=1).mean()
                correcut_num1,sample_num1=calculate_accuracy_total(pred_topk[:, 0], lbl_task.view(-1).detach().cpu().numpy())
                total_correct_num1 +=correcut_num1
                total_sample_num1 +=sample_num1
                top2_correct = (pred_topk[:, :2] == lbl_task.view(-1).detach().cpu().numpy()[:, None]).any(axis=1)
                correct_num2 = np.sum(top2_correct)
                sample_num2 = len(top2_correct)
                total_correct_num2 += correct_num2
                total_sample_num2 += sample_num2
                
                top3_correct = (pred_topk[:, :3] == lbl_task.view(-1).detach().cpu().numpy()[:, None]).any(axis=1)
                correct_num3 = np.sum(top3_correct)
                sample_num3 = len(top3_correct)
                total_correct_num3 += correct_num3
                total_sample_num3 += sample_num3         
                print(f"Accuracy (Top-1): {accuracy}")
                print(f"Accuracy (Top-2): {top2_accuracy}")
                print(f"Accuracy (Top-3): {top3_accuracy}")
                
                print("predArray, actualArray: ", pred_topk[:, 0], lbl_task.view(-1).detach().cpu().numpy())
            else:
                raise ValueError("Invalid task specified. Use 0, 1, 2, or 3.")
    
     
            pred_top1 = pred_scores.topk(1, dim=-1).indices.squeeze(-1).detach().cpu().numpy()
            lbl_np = lbl_task.view(-1).detach().cpu().numpy()
    
            mseVal = ((pred_top1.astype(np.float32) - lbl_np) ** 2).mean()
            numInputs = lbl_task.size(0)
            totalMSE.update(mseVal, numInputs)
            print(f"Current totalMSE: {totalMSE.avg:.4f}")
    
            desName = batch.desName
            batchData.append([pred_top1, lbl_np, desName])
    
            for pred, actual in zip(pred_top1, lbl_np):
                actual = int(actual)
                pred = int(pred)
                if pred != actual:
                    if actual not in incorrect_counts:
                        incorrect_counts[actual] = np.zeros(num_classes)
                    incorrect_counts[actual][pred] += 1
    
    incorrect_distribution = {}
    for actual, counts in incorrect_counts.items():
        total_incorrect = counts.sum()
        if total_incorrect > 0:
            incorrect_distribution[actual] = counts / total_incorrect
    
    for actual, distribution in incorrect_distribution.items():
        print(f"Actual class {actual} incorrect prediction distribution: {distribution}")
    
    if task == 1:
       overall_accuracy_top1 =  total_correct_num0 / total_sample_num0 
       print(f"Overall Accuracy (Top-1): {overall_accuracy_top1:.4f}")
    if task ==2:
       overall_accuracy_top1 =  total_correct_num1 / total_sample_num1 
       print(f"Overall Accuracy (Top-1): {overall_accuracy_top1:.4f}")  
       overall_accuracy_top2 =  total_correct_num2 / total_sample_num2 
       print(f"Overall Accuracy (Top-2): {overall_accuracy_top2:.4f}")
       overall_accuracy_top3 =  total_correct_num3 / total_sample_num3 
       print(f"Overall Accuracy (Top-3): {overall_accuracy_top3:.4f}")
    return totalMSE.avg, batchData, incorrect_distribution


def arc_inference_model1(model1, device, dataloader1, filepath_model1):
    model1.eval()

    with torch.no_grad():
        # Open file for writing predictions
        with open(filepath_model1, 'w') as file1:
            # Process for model 1
            for batch in tqdm(dataloader1, desc="Model 1 Iteration", file=sys.stdout):
                batch = batch.to(device)

                # Access design names directly from batch
                design_names = batch.desName
                
                # Model 1 predictions
                pred_task1, _, _ = model1(batch, 1)
                pred_scores1 = pred_task1
                pred_top1_1 = pred_scores1.argmax(dim=-1).detach().cpu().numpy()

                # Write to file for model 1
                for name, pred in zip(design_names, pred_top1_1):
                    file1.write(f"{name},{pred}\n")
def arc_inference_model2(model2, device, dataloader2, filepath_predictions, filepath_times):
    model2.eval()

    with torch.no_grad():
        with open(filepath_predictions, 'w') as prediction_file, open(filepath_times, 'w') as time_file:
            for batch in tqdm(dataloader2, desc="Model 2 Iteration", file=sys.stdout):
                start_time = time.time()
                
                batch = batch.to(device)
                design_names = batch.desName
                
                _, _, pred_task3 = model2(batch, 2)
                pred_scores2 = pred_task3
                
                top3_scores, top3_indices = torch.topk(pred_scores2, k=3, dim=-1)
                top3_indices = top3_indices.detach().cpu().numpy()

                end_time = time.time()
                inference_time = (end_time - start_time) * 2  # Multiply by 2 as per your requirement

                for name, preds in zip(design_names, top3_indices):
                    pred_str = ",".join(map(str, preds))
                    prediction_file.write(f"{name},{pred_str}\n")
                    time_file.write(f"{name},{inference_time:.4f}\n")
# def evaluate_plot(model, device, dataloader, task, num_classes):
#     model.eval()
#     batchData = []
#     incorrect_counts = {}
    
#     correct_top1 = 0
#     correct_top2 = 0
#     total = 0
    
#     with torch.no_grad():
#         for _, batch in enumerate(tqdm(dataloader, desc="Iteration", file=sys.stdout)):
#             batch = batch.to(device)
#             pred_task0, pred_task1, pred_task2, pred_task3 = model(batch, task)
#             lbl = batch.y.float()
            

#             if task == 1:
#                 lbl_task = lbl[:, 1].long() - 1
#                 pred_scores = pred_task1
#                 accuracy = calculate_accuracy(pred_top1, lbl_task.view(-1).detach().cpu().numpy())
#                 print(f"Accuracy (Top-1): {accuracy}")
#             elif task == 2:
#                 lbl_task = lbl[:, 3].long() - 1
#                 pred_scores = pred_task2
#                 accuracy = calculate_accuracy(pred_top1, lbl_task.view(-1).detach().cpu().numpy())
#                 print(f"Accuracy (Top-1): {accuracy}")
#             elif task == 3:
#                 lbl_task = lbl[:, 2].long() - 1
#                 pred_scores = pred_task3
#                 topk = [1, 2, ]
                
#                 print(f"Accuracy (Top-1): {accuracy}")
#                 print(f"Accuracy (Top-2): {top2_accuracy}")
#                 print(f"Accuracy (Top-3): {top3_accuracy}")
                
#                 print("predArray, actualArray: ", pred_topk[:, 0], lbl_task.view(-1).detach().cpu().numpy())
#             else:
#                 raise ValueError("Invalid task specified. Use 0, 1, 2, or 3.")
    
     
#             pred_top1 = pred_scores.topk(1, dim=-1).indices.squeeze(-1).detach().cpu().numpy()
#             lbl_np = lbl_task.view(-1).detach().cpu().numpy()
    
#             mseVal = ((pred_top1.astype(np.float32) - lbl_np) ** 2).mean()
#             numInputs = lbl_task.size(0)
#             totalMSE.update(mseVal, numInputs)
#             print(f"Current totalMSE: {totalMSE.avg:.4f}")
    
#             desName = batch.desName
#             batchData.append([pred_top1, lbl_np, desName])
    
#             for pred, actual in zip(pred_top1, lbl_np):
#                 actual = int(actual)
#                 pred = int(pred)
#                 if pred != actual:
#                     if actual not in incorrect_counts:
#                         incorrect_counts[actual] = np.zeros(num_classes)
#                     incorrect_counts[actual][pred] += 1
    
#     incorrect_distribution = {}
#     for actual, counts in incorrect_counts.items():
#         total_incorrect = counts.sum()
#         if total_incorrect > 0:
#             incorrect_distribution[actual] = counts / total_incorrect
    
#     for actual, distribution in incorrect_distribution.items():
#         print(f"Actual class {actual} incorrect prediction distribution: {distribution}")
    

    
#     return totalMSE.avg, batchData, incorrect_distribution

def plotChart(x,y,xlabel,ylabel,leg_label,title,DUMP_DIR):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(x,y, label=leg_label)
    leg = plt.legend(loc='best', ncol=2, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.xlabel(xlabel, weight='bold')
    plt.ylabel(ylabel, weight='bold')
    plt.title(title,weight='bold')
    plt.savefig(osp.join(DUMP_DIR,title+'.png'), format='png', bbox_inches='tight')

def calculate_accuracy(predictions, actuals):
    correct_positions = np.sum(predictions == actuals)
    total_positions = len(actuals)

    if total_positions == 0:
        return 0.0
        
    return correct_positions / total_positions
def calculate_accuracy_total(predictions, actuals):
    correct_positions = np.sum(predictions == actuals)
    total_positions = len(actuals)

    # if total_positions == 0:
    #     return 0.0
        
    return correct_positions,total_positions
def extract_graph_features(edge_index):
    G = nx.Graph()
    G.add_edges_from(edge_index.t().tolist())
    
    density = nx.density(G)
    
    cluster_coefficient = nx.average_clustering(G)
    
    num_nodes = G.number_of_nodes()
    avg_degree = sum(dict(G.degree()).values()) / num_nodes if num_nodes > 0 else 0
    
    return torch.tensor([density, cluster_coefficient, avg_degree], dtype=torch.float)



def load_graph_features(output_dir):
    if os.path.exists(os.path.join(output_dir, 'graph_features_dict.pt')):
        graph_features_dict = torch.load(os.path.join(output_dir, 'graph_features_dict.pt'))
        return graph_features_dict
    return None






