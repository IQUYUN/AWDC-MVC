import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "3"
import torch
import torch.nn.functional as F
from network import Network
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import ContrastiveLoss
from loss import LableLoss
from dataloader import load_data
from metricfinal import validfinal

Dataname = 'Cifar100'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--learning_rate", default=0.0001)
parser.add_argument("--weight_decay", default=0.0)
parser.add_argument("--pre_epochs", default=100)
parser.add_argument("--con_epochs", default=200)
parser.add_argument("--feature_dim", default=256)
parser.add_argument("--high_feature_dim", default=32)
parser.add_argument("--temperature1", default=1.0)
parser.add_argument("--temperature2", default=1.0)
parser.add_argument("--param1", default=1)
parser.add_argument("--param2", default=1)
parser.add_argument("--param3", default=1)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preserve CLI-provided params and provide an env-guard to respect them after dataset presets.
_cli_param1, _cli_param2, _cli_param3 = args.param1, args.param2, args.param3
_force_params = os.environ.get("MVC_FORCE_PARAMS", "")
_cli_temp1, _cli_temp2 = args.temperature1, args.temperature2
_force_temps = os.environ.get("MVC_FORCE_TEMPS", "")

if args.dataset == "CCV":
    # 聚类头8个，2层 
    args.pre_epochs = 50
    args.con_epochs = 600
    args.learning_rate = 0.0003
    args.temperature1 = 0.35
    args.temperature2= 0.7
    args.param1=0.001
    args.param2=0.0001
    args.param3=1
    seed = 102
if args.dataset == "Cifar100":
    args.pre_epochs = 100
    args.con_epochs = 200
    args.learning_rate = 0.0003
    args.temperature1 = 0.5
    args.temperature2= 0.3
    args.param1=0.1
    args.param2=0.01
    args.param3=0.1
    seed = 10
if args.dataset == "Prokaryotic":
    # 聚类头8个，2层 
    args.pre_epochs = 50
    args.con_epochs = 201
    args.learning_rate = 0.00001
    args.temperature1 = 0.8
    args.temperature2= 0.3
    args.param1=0.4
    args.param2=0.4
    args.param3=0.001
    seed = 10008

# If requested, force back CLI-provided param weights after dataset presets.
if _force_params:
    try:
        args.param1 = float(_cli_param1)
        args.param2 = float(_cli_param2)
        args.param3 = float(_cli_param3)
    except Exception:
        pass
if _force_temps:
    try:
        args.temperature1 = float(_cli_temp1)
        args.temperature2 = float(_cli_temp2)
    except Exception:
        pass

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(seed)

dataset, dims, view, data_size, class_num = load_data(args.dataset)
data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
data_loader_all = torch.utils.data.DataLoader(
    dataset, 
    batch_size=data_size, 
    shuffle=True,                                                
    drop_last=False)

def compute_infoNCE_weights(rs, H, view):
    N = H.shape[0]
    weights = []

    for v in range(view):
        r = rs[v]
        logits = torch.matmul(r, H.T)   # [N x N]
        
        labels = torch.arange(N).to(r.device)  # 正样本对：同索引
        loss = F.cross_entropy(logits, labels)  # InfoNCE loss

        mi = -loss  # 互信息估计（越大越好）
        weights.append(mi)
    weights = torch.stack(weights)
    weights = torch.softmax(weights, dim=0)  # 转为加权权重
    return weights


def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        # 清楚上一次梯度
        optimizer.zero_grad()
        # xrs是重构结果，预训练只需要重构结果即可
        xrs, zs, rs, H ,_= model(xs)
        # 计算损失
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        # 计算当前梯度
        loss.backward()
        # 使用梯度更新参数
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))

def contrastive_train(epoch):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, zs, rs, H,weighting = model(xs)

        # 基于概率的对比标签
        ss,hp = model.calculate_s(xs)

        loss_list = []

        # 每个视图基于信息熵的权重
        with torch.no_grad():
            weight = compute_infoNCE_weights(rs, H, view)

        for v in range(view):
            # Self-weighted contrastive learning loss
            # 原来的特征级别的对比函数
            loss_list.append(contrastiveloss(H, rs[v],weighting[v])*args.param1)
            # 现在的概率级别的对比函数
            loss_list.append(lableloss.forward_label(ss[v],hp,weighting[v])*args.param2)
            # Reconstruction loss
            loss_list.append(mse(xs[v], xrs[v]))
        # 计算权重约束损失
        loss_weight_consistency = F.mse_loss(weighting, weight)*args.param3
        loss_list.append(loss_weight_consistency)
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))


accs = []
nmis = []
purs = []
if not os.path.exists('./models'):
    os.makedirs('./models')
T = 1
for i in range(T):
    print("ROUND:{}".format(i+1))
    setup_seed(seed)
    model = Network(view, dims, args.feature_dim, args.high_feature_dim,class_num, device)
    print(model)
    model = model.to(device)
    state = model.state_dict()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    contrastiveloss = ContrastiveLoss(args.batch_size, args.temperature1, device).to(device)
    lableloss=LableLoss(args.batch_size, class_num,args.temperature2, device).to(device)
    best_acc, best_nmi, best_pur = 0, 0, 0

    epoch = 1
    while epoch <= args.pre_epochs:
        pretrain(epoch)
        epoch += 1
    # acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=True, epoch=epoch)

    while epoch <= args.pre_epochs + args.con_epochs:
        contrastive_train(epoch)
        #acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=False, epoch=epoch)
        acc, nmi, ari, pur = validfinal(model, device, dataset, view, data_size, epoch)

        if acc > best_acc:
            best_acc, best_nmi, best_pur = acc, nmi, pur
            state = model.state_dict()
            torch.save(state, './models/' + args.dataset + '.pth')
        epoch += 1

    # The final result
    accs.append(best_acc)
    nmis.append(best_nmi)
    purs.append(best_pur)
    print('The best clustering performace: ACC = {:.4f} NMI = {:.4f} PUR={:.4f}'.format(best_acc, best_nmi, best_pur))

