# -*- encoding: utf-8 -*-
'''
@File    :   modules.py
@Time    :   2021/04/17 13:59:17
@Author  :   Jian HU 
@Version :   0.01
@Contact :   hujian@mail.ustc.edu.cn
'''

import os
import numpy as np
import torch
from random import shuffle
import torch
import torch.nn.functional as F
from cerebra.loss_init import comp_initloss
from cerebra.loss import comp_loss
from cerebra.loss_finetuning import comp_finetuningloss
from cerebra.loadata import *

from cerebra.config import model_config
from cerebra.humodel.model import AlphaFold
from torch.cuda.amp import autocast, GradScaler 

os.environ['CUDA_VISIBLE_DEVICES']='2'
gpu0 = torch.device('cuda:0')

torch.set_num_threads(20)


def dataloader(pdb_list, data_dir, mode='train'):
    if mode == 'train':
        cycle_num = (torch.rand(1) * 4).long() + 1
        cycle_num = torch.clip(cycle_num, min=1, max=4).item()
    
    batch_feats = {}
    msa_feat, bert_mask, true_msa = [], [], []
    for pdb in pdb_list[0]:
        data_file = f'{data_dir}{pdb}.h5'
        single_feat = get_feature_extend(data_file, cycle_num, max_crop=pdb_list[1], mode=mode)
        for f in single_feat.keys():
            if f == 'msa_feat':
                msa_feat.append(single_feat['msa_feat'])
            elif f == 'bert_mask':
                bert_mask.append(single_feat['bert_mask'])
            elif f == 'true_msa':
                true_msa.append(single_feat['true_msa'])
            else:
                if f in batch_feats:
                    tmp = batch_feats[f]
                    tmp.append(single_feat[f])
                    batch_feats[f] = tmp
                else:
                    tmp = [single_feat[f]]
                    batch_feats[f] = tmp
    min_seq_num = min([x.shape[1] for x in msa_feat])
    batch_feats['msa_feat'] = [msa[:, :min_seq_num] for msa in msa_feat]
    batch_feats['bert_mask'] = [mask[:min_seq_num] for mask in bert_mask]
    batch_feats['true_msa'] = [msa[:min_seq_num] for msa in true_msa]
    feats_preload = {}
    for k, v in batch_feats.items():
        if k == 'seq':
            feats_preload[k] = v
        else:
            v = torch.stack(v)
            if k in ['target_feat', 'residue_index', 'msa_feat']:
                v = v.transpose(1, 0)
            feats_preload[k] = v
    return feats_preload

def split_minibatch(train_list):
    params = {
        96:2,
        128:4,
    }

    max_length, min_length = max(params.keys()), min(params.keys())
    cutoffs = np.sort(np.array(list(params.keys())))
    length_collection = {}
    for idx in params.keys():
        length_collection[idx] = []

    length_dict = {}
    with open(train_list, 'r') as f:
        for l in f.readlines():
            tmp = l.strip().split()
            pdb, length = tmp[0], int(tmp[-1])
            length_dict[pdb] = length

            if length <= min_length:
                tmp = length_collection[min_length]
                tmp.append(pdb)
                length_collection[min_length] = tmp
            elif length > max_length:
                tmp = length_collection[max_length]
                tmp.append(pdb)
                length_collection[max_length] = tmp
            else:
                for c in range(1, len(cutoffs)):
                    if cutoffs[c-1] < length <= cutoffs[c]:
                        tmp = length_collection[cutoffs[c]]
                        tmp.append(pdb)
                        length_collection[cutoffs[c]] = tmp

    mini_batch = []
    for length_cutoff, batch_size in params.items():
        lc = length_collection[length_cutoff]
        shuffle(lc)
        shuffle(lc)
        shuffle(lc)
        mini_batch.extend([[lc[i:i+batch_size], length_cutoff] for i in range(0, len(lc), batch_size)])
    shuffle(mini_batch)
    shuffle(mini_batch)
    shuffle(mini_batch)
    mini_batch_maxcrop = []
    for pdbs, _ in mini_batch:
        min_length = []
        for p in pdbs:
            min_length.append(length_dict[p])
        min_length = min(min_length)
        if min_length > max_length:
            mini_batch_maxcrop.append([pdbs, max_length])
        else:
            mini_batch_maxcrop.append([pdbs, min_length])
    return mini_batch_maxcrop

def run_cycle(batch, LR, step, accumulation_steps):
    m_1_prev, z_prev, x_prev, pLDDT_prev = None, None, None, None
    prevs = [m_1_prev, z_prev, x_prev, pLDDT_prev]

    is_grad_enabled = torch.is_grad_enabled()
    dims = batch["msa_feat"].shape

    num_iters = dims[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    length = dims[3]
    k = 32
    # if length < 96:
    #     k = 12
    AncherList = np.array([(x+1) * length/(k + 1) for x in range(k)]) + np.random.randint(round(length/k), size=(k))
    AncherList = AncherList.astype(int)
    AncherList = np.clip(AncherList, a_min=1, a_max=length-2)

    for cycle_no in range(num_iters): 
        is_final_iter = cycle_no == (num_iters - 1)
        feats = {}

        feats["seq_mask"] = torch.ones([dims[1], dims[3]], dtype=torch.float32)
        feats["msa_mask"] = torch.ones([dims[1], dims[2], dims[3]], dtype=torch.float32)

        feats['target_feat'] = batch['target_feat'][cycle_no]
        feats['residue_index'] = batch['residue_index'][cycle_no]
        feats['msa_feat'] = batch['msa_feat'][cycle_no]
        feats['esm2'] = batch['esm2']

        with autocast():
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                dtype = next(model.parameters()).dtype
                device = next(model.parameters()).device
                for k in feats:
                    if(feats[k].dtype == torch.float32):
                        feats[k] = feats[k].to(dtype=dtype)
                    feats[k] = feats[k].to(device)

                if is_final_iter:
                    if torch.is_autocast_enabled():
                        torch.clear_autocast_cache()

                m_1_prev, z_prev, x_prev, pLDDT_prev, outputs = model(feats, prevs, AncherList, _recycle=(num_iters > 1))
                if is_final_iter:
                    loss, loss_items  = comp_initloss(outputs, batch, AncherList)
                    # loss, loss_items  = comp_loss(outputs, batch, AncherList)
                    # loss, loss_items  = comp_finetuningloss(outputs, batch, AncherList)
                else:
                    prevs = [m_1_prev.detach(), z_prev.detach(), x_prev.detach(), pLDDT_prev.detach()]

    print(loss)
    loss = loss/accumulation_steps      
    scaler.scale(loss).backward()
    if step % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    return loss_items


low_prec = True
config = model_config(
    name="model_3",
    train=True, 
    low_prec=low_prec,
)

model = AlphaFold(config)
# model.load_state_dict(torch.load('/home/hujian/Model2Sidechain/checkpoint/model0.pt', map_location='cpu'))
model = model.to(gpu0)

step = 0
scaler = GradScaler()
for epoch in range(3):
    accumulation_steps = 1
    train_log = open('train.log', 'a')

    """
    comp_initloss: step;T_consistence;T_consistence;dist;Dihedral;quaternion;translation;plddt;MSA;unscalingloss;loss_sum;avglddt;bestlddt;lr
    loss_items   : step;T_consistence;T_consistence;dist;Dihedral;quaternion;translation;plddt;MSA;unscalingloss;loss_sum;avglddt;bestlddt;lr
    loss_items   : step;T_consistence;T_consistence;dist;Dihedral;quaternion;translation;plddt;MSA;FAPE;angle;viol;unscalingloss;loss_sum;avglddt;bestlddt;lr

    """

    train_log.write(f"EPOCH: {epoch}\taccumulation_steps={accumulation_steps}\n")
    data_dir = 'data/'
    train_list = 'list.txt'
    train_tmp = split_minibatch(train_list)
    for i in train_tmp:

        step += 1

        LR = 1e-6 * step
        LR = min(LR, 1e-3)

        batch = dataloader(i, data_dir)
        loss_items = run_cycle(batch, LR, step, accumulation_steps)
        train_log.write(f'step {step}\t')
        for i in loss_items:
            train_log.write('%.4f\t' % (i))
        train_log.write('%.7f\n' % LR)
        train_log.flush()
    torch.save(model.state_dict(), './checkpoint/model{}.pt'.format(epoch))
    train_log.close()
