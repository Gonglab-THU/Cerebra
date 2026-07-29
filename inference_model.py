# -*- encoding: utf-8 -*-
'''
@File    :   modules.py
@Time    :   2021/04/17 13:59:17
@Author  :   Jian HU 
@Version :   0.01
@Contact :   hujian@mail.ustc.edu.cn
'''

import os, sys
import json
import numpy as np
import torch

# from train_utils import EMA
# from loss_mask import comp_loss

from functools import reduce
from random import shuffle
import random
import torch
import torch.nn.functional as F
from cerebra_model.np import residue_constants as rc


from data_nomask import *

from torch.cuda.amp import autocast

from cerebra_model.config import model_config
from cerebra_model.humodel.model_with_conf import AlphaFold
from cerebra_model.humodel.structure_module import NormQuaternion, NormQuaternionMM
from sklearn.cluster import KMeans
import esm
from cerebra_model.get_all_atoms_new import  hu_model_pred_to_atom14_pos,make_atom14_masks
from cerebra_model.np.protein import*
from cerebra_model.utils.tensor_utils import (
    tree_map,
    tensor_tree_map,
    batched_gather,
)

torch.set_num_threads(20)

def setup_seed(seed):
    os.environ['PYTHONSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

setup_seed(42)

def bf16_autocast(device, enabled=False):
    device = torch.device(device)
    return autocast(enabled=(enabled and device.type == 'cuda'), dtype=torch.bfloat16)

def get_esm2(seq,name,esm_model, batch_converter, device):

    print(seq,name)
    with torch.no_grad():
            data = [[name,seq]]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            results = esm_model(batch_tokens.to(device), repr_layers=[36], return_contacts=True)
            
            token_embeds = results["representations"][36] # (batch=1, L+2, dim=2560) token_representations
            token_embeds = token_embeds[:, 1:-1, :].to(dtype=torch.float32).cpu().detach() # (batch=1, L, dim=2560)


            # print(results["attentions"].shape, results["contacts"].shape)
            ###### attention map and contact map ######
            attentions = results["attentions"] #(batch, layers=36, heads=40, L+2, L+2)
            attentions = attentions[:, -1, :, 1:-1, 1:-1] #(40, L, L)
            contacts = results["contacts"] # (1, L, L)
            # feature_2D = torch.cat([attentions, contacts],dim=0).unsqueeze(0).to(dtype=torch.float32).cpu().detach().numpy() # (1, 41, L, L)

            feature_2D = torch.cat([contacts.unsqueeze(1), attentions], dim=1).cpu().detach().numpy()

            for idx, v in enumerate(data):
                pdb = v[0]
                # print(pdb)
                length = len(v[1])
                X1D = token_embeds[idx][:length]
                X2D = feature_2D[idx][:, :length, :length]

                return X1D
def single_pdb_write_IO(pred_all_atoms,plddt,batch,path):
    num_batch,L  = pred_all_atoms.shape[:2]
    pred_all_atoms =  batched_gather(
            pred_all_atoms,
            batch['residx_atom37_to_atom14'],
            dim=-2,
            no_batch_dims=len(pred_all_atoms.shape[:-2]),
        )

        
    for i in range(num_batch):
            pdb_write = {}
            pdb_write['residue_index'] = np.arange(L)
            pdb_write["aatype"] = batch['aatype'][i].cpu().numpy()
            pdb_write["final_atom_positions"] = pred_all_atoms[i].cpu().numpy()
            pdb_write["final_atom_mask"] = batch["atom37_atom_exists"][i].cpu().numpy()
            b_factors = plddt[i].unsqueeze(-1).repeat(1,37)*100
            b_factors = torch.clip(b_factors,min = 0,max=99.99).cpu().numpy()
            protein1 = from_prediction(pdb_write,pdb_write,b_factors)
            with open(f'{path}', 'w') as fp:
                fp.write(to_pdb(protein1))
def compute_plddt(logits: torch.Tensor) -> torch.Tensor:
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width, end=1.0, step=bin_width, device=logits.device
    )
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_lddt_ca = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return pred_lddt_ca          
def pdb_file_out(pred_output, batch, Ancher_id, use_comb=False):
    device = pred_output['translation'][-1].device
    if use_comb:
        xyz = pred_output['translation'][-1][0].detach().cpu().numpy()
        quaternion = pred_output['quaternion'][-1]
        main_anchor_pos = xyz[Ancher_id]
        translations, rotations = [], []

        for i in range(xyz.shape[0]):
            t, r = getTransformation(main_anchor_pos, xyz[i], return_rotation=True)
            translations.append(t)
            rotations.append(Rotation2Quaternion(torch.tensor(r, device=device)))

        translations = torch.tensor(np.array(translations), device=device).permute(1, 0, 2)
        consensus_translations = torch.stack([_compute_consensus_positions(t) for t in translations])

        rotations = torch.stack(rotations)[None, :, None, :]
        rotations = rotations.to(device=quaternion.device, dtype=quaternion.dtype)
        comb_q = NormQuaternionMM(rotations, quaternion)
        comb_q = comb_q[0].permute(1, 0, 2)
        consensus_rotations = torch.stack([_compute_consensus_positions(q) for q in comb_q])

        quaternion = NormQuaternion(consensus_rotations.unsqueeze(0))
        translation = consensus_translations.unsqueeze(0)
    else:
        quaternion = pred_output['quaternion'][-1][:,Ancher_id]
        translation = pred_output['translation'][-1][:,Ancher_id]
    angles = pred_output['angles'][-1].to(device)
    # print('plddt',pred_output['pLDDT'].shape)
    print('plddt',pred_output['pLDDT'].shape)
    plddt = pred_output['pLDDT']
    # plddt = compute_plddt(pred_output['pLDDT'])
    # print('plddt',plddt.shape)
    # plddt = torch.mean(plddt,dim=-1)
    # print('plddt',plddt.shape)
    # # print(pred_output['pLDDT'][0,0])
    plddt = torch.mean(plddt,dim=1) #[b

    raw_seq = batch['true_msa'][0,0]
    of_seq =[]
    # print('quaternion',quaternion.shape)
    # print('translation',translation.shape)
    # print('angles',angles.shape)
    # print('plddt',plddt.shape) 
    # print('raw_seq',raw_seq.shape)
    for i in raw_seq:
        of_seq.append(rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE[i])
    of_seq = torch.LongTensor(of_seq).unsqueeze(0).to(device)
    # print('of_seq',of_seq.shape)
    all_atom_pos_14 = hu_model_pred_to_atom14_pos(quaternion,translation,angles,of_seq)
    batch['aatype'] = of_seq
    batch = make_atom14_masks(batch)
        
    return all_atom_pos_14, plddt,batch
    
def get_fasta_feature(fasta,cycle_num,esm_model,batch_converter,device):
    mode = 'eval'
    HHBLITS_AA_TO_ID = {
        "A": 0, "B": 2, "C": 1, "D": 2, "E": 3,
        "F": 4, "G": 5, "H": 6, "I": 7, "J": 20,
        "K": 8, "L": 9, "M": 10, "N": 11, "O": 20,
        "P": 12, "Q": 13, "R": 14, "S": 15, "T": 16,
        "U": 1, "V": 17, "W": 18, "X": 20, "Y": 19,
        "Z": 3, "-": 21
    }

    msa = []
    idx_target = 0
 
    seq = []
    for AA in list(fasta):
        if ord(AA) < 97:
            if AA in HHBLITS_AA_TO_ID.keys():
                seq.append(HHBLITS_AA_TO_ID[AA])
            else:
                seq.append(20)
    msa.append(seq)
    msa = torch.LongTensor(msa)

    esm2 = get_esm2(fasta,'inference',esm_model,batch_converter,device)

    if msa.shape[0] > 2000:
        ratio = (msa < 21).float().sum(1)/msa.shape[1]
        tmp = []
        for idx, gap_ratio in enumerate(ratio):
            if gap_ratio > 0.25:
                tmp.append(msa[idx])
        msa = torch.stack(tmp)

    pdb_length = msa.shape[1]
    c1, c2 = 0, pdb_length

    nonensemble_feat = make_msa_features(msa)
    nonensemble_feat = make_hhblits_profile(nonensemble_feat)
    nonensemble_feat = make_msa_mask(nonensemble_feat)

    max_seq = 256
    msa_feat, target_feat = [], []
    msa_depth = []
    for cycle in range(cycle_num):
        
        
        msa_features = sample_msa(nonensemble_feat.copy(), max_seq, seed_num+cycle)
        msa_depth.append(min(1, 128) / 128.0)
        msa_features = make_masked_msa(msa_features, mode)
        msa_features = nearest_neighbor_clusters(msa_features)
        msa_features = summarize_clusters(msa_features)
        msa_features = make_msa_feat(msa_features)
        msa_feat.append(msa_features['msa_feat'])
        target_feat.append(msa_features['target_feat'])
    
    
    msa_feat = torch.stack(msa_feat)
    true_msa = msa_features['true_msa']
    bert_mask = msa_features['bert_mask']

    msa_num, seq_length = true_msa.shape 

    max_seq = 128
    if msa_num < max_seq and mode == 'eval':
        repeat_num = int(max_seq/msa_num) + 1
        msa_feat = msa_feat[:, None, :, :, :].repeat(1, repeat_num, 1, 1, 1).reshape(cycle_num, -1, seq_length, 49)[:, :max_seq]
        true_msa = true_msa[None].repeat(repeat_num, 1, 1).reshape(-1, seq_length)[:max_seq]
        bert_mask = bert_mask[None].repeat(repeat_num, 1, 1).reshape(-1, seq_length)[:max_seq]

    ret_feature = {}
    ret_feature['msa_feat'] = msa_feat
    ret_feature['true_msa'] = true_msa
    ret_feature['bert_mask'] = bert_mask
    ret_feature['esm2'] = esm2
    ret_feature['msa_depth'] = torch.tensor(msa_depth, dtype=torch.float32)
    ret_feature['target_feat'] = torch.stack(target_feat)
    ret_feature['residue_index'] = torch.arange(c1, c2)[None].repeat(cycle_num, 1)
    ret_feature['seq'] = fasta
    mask = torch.ones(seq_length)
    ret_feature['mask'] = mask
    return ret_feature
def dataloader(pdb, fasta,esm_model, batch_converter,device):
    mode='eval'
    if mode == 'eval':
        cycle_num = 4
    
    batch_feats = {}
    msa_feat, bert_mask, true_msa = [], [], []
    for pdb in [pdb]:
        single_feat = get_fasta_feature(fasta, cycle_num, esm_model, batch_converter,device)
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

def run_cycle(batch,model,use_bf16=False):
    m_1_prev, z_prev, x_prev, pLDDT_prev = None, None, None, None
    prevs = [m_1_prev, z_prev, x_prev]

    dims = batch["msa_feat"].shape
    num_iters = dims[0]
    length = dims[3]

    n_clusters = 24
    # if length < 96:
    #     n_clusters = 12
    if 324 >= length >= 224:
        n_clusters = 32
    if length > 324:
        n_clusters = 48
    if length > 500:
        n_clusters = 56

    
    plddt_collection, outputs_collection = [], []
    AncherList = np.array([int((x + 1) * length/(n_clusters + 1)) for x in range(n_clusters)])
    AncherList = np.array([int(x* (length - 8)/n_clusters) for x in range(n_clusters)]) + 5
    # AncherList = np.array([int(x* (length - 8)/(n_clusters+1)) for x in range(n_clusters)]) + 5
    # AncherList = np.array([int(length/(n_clusters + 1)) * (x + 1) for x in range(n_clusters)]) #+ np.random.randint(round(length/n_clusters), size=(n_clusters))
    AncherList = AncherList.astype(int)
    AncherList = np.clip(AncherList, a_min=2, a_max=length-2)

    for cycle_no in range(num_iters):
        # print(AncherList)
        is_final_iter = cycle_no == (num_iters - 1)
        feats = {}

        feats["seq_mask"] = torch.ones([dims[1], dims[3]], dtype=torch.float32)
        feats["msa_mask"] = torch.ones([dims[1], dims[2], dims[3]], dtype=torch.float32)

        feats['target_feat'] = batch['target_feat'][cycle_no]
        feats['residue_index'] = batch['residue_index'][cycle_no]

        feats['msa_feat'] = batch['msa_feat'][cycle_no]
        msa_target = feats['msa_feat'][0, 0, :, :23]
        #feats['msa_feat'] = torch.zeros_like(batch['msa_feat'][cycle_no])
        #feats['msa_feat'][:, :, :, :23] = msa_target

        feats['esm2'] = batch['esm2']
        # feats['esm2'] = torch.zeros_like(batch['esm2'])

        with torch.no_grad():
            dtype = next(model.parameters()).dtype
            device = next(model.parameters()).device
            for k in feats:
                if(feats[k].dtype == torch.float32):
                    feats[k] = feats[k].to(dtype=dtype)
                feats[k] = feats[k].to(device)
            
            # print(feats['msa_feat'].shape)

            with bf16_autocast(device, enabled=use_bf16):
                m_1_prev, z_prev, x_prev, outputs = model(feats, prevs, AncherList, _recycle=(num_iters > 1))

            outputs_collection.append(outputs)
            if is_final_iter:
                return outputs_collection[-1]
            else:
                prevs = [m_1_prev.detach(), z_prev.detach(), x_prev.detach()]
                # AncherList = select_Ancher(x_prev.detach().cpu(), outputs['pLDDT'].detach().cpu(), n_clusters)




    return outputs_collection[-1]

def Rotation2Quaternion(r, eps=1e-6):
    r00, r11, r22 = r[..., 0, 0], r[..., 1, 1], r[..., 2, 2]
    trace = r00 + r11 + r22
    safe_mask = (trace + 1.0) > eps
    q = torch.zeros(*r.shape[:-2], 4, device=r.device, dtype=r.dtype)
    
    if safe_mask.any():
        sub_r = r[safe_mask]
        sub_trace = trace[safe_mask]
        a = torch.sqrt(sub_trace + 1.0) * 0.5
        denom = 4.0 * a
        b = (sub_r[:, 2, 1] - sub_r[:, 1, 2]) / denom
        c = (sub_r[:, 0, 2] - sub_r[:, 2, 0]) / denom
        d = (sub_r[:, 1, 0] - sub_r[:, 0, 1]) / denom
        q[safe_mask] = torch.stack([a, b, c, d], dim=-1)

    unsafe_mask = ~safe_mask
    if unsafe_mask.any():
        sub_r = r[unsafe_mask]
        sub_r00 = sub_r[:, 0, 0]
        sub_r11 = sub_r[:, 1, 1]
        sub_r22 = sub_r[:, 2, 2]
        t1 = 1.0 + sub_r00 - sub_r11 - sub_r22
        t2 = 1.0 - sub_r00 + sub_r11 - sub_r22
        t3 = 1.0 - sub_r00 - sub_r11 + sub_r22
        candidates = torch.stack([t1, t2, t3], dim=-1)
        vals, idx = torch.max(candidates, dim=-1)
        t = torch.sqrt(torch.relu(vals)) 
        t_inv = 0.5 / (t + 1e-8)
        t = 0.5 * t
        sub_q = torch.zeros(sub_r.shape[0], 4, device=r.device, dtype=r.dtype)
        
        mask_x = (idx == 0)
        if mask_x.any():
            sub_q[mask_x, 0] = (sub_r[mask_x, 2, 1] - sub_r[mask_x, 1, 2]) * t_inv[mask_x]
            sub_q[mask_x, 1] = t[mask_x]
            sub_q[mask_x, 2] = (sub_r[mask_x, 1, 0] + sub_r[mask_x, 0, 1]) * t_inv[mask_x]
            sub_q[mask_x, 3] = (sub_r[mask_x, 0, 2] + sub_r[mask_x, 2, 0]) * t_inv[mask_x]

        mask_y = (idx == 1)
        if mask_y.any():
            sub_q[mask_y, 0] = (sub_r[mask_y, 0, 2] - sub_r[mask_y, 2, 0]) * t_inv[mask_y]
            sub_q[mask_y, 1] = (sub_r[mask_y, 1, 0] + sub_r[mask_y, 0, 1]) * t_inv[mask_y]
            sub_q[mask_y, 2] = t[mask_y]
            sub_q[mask_y, 3] = (sub_r[mask_y, 2, 1] + sub_r[mask_y, 1, 2]) * t_inv[mask_y]

        mask_z = (idx == 2)
        if mask_z.any():
            sub_q[mask_z, 0] = (sub_r[mask_z, 1, 0] - sub_r[mask_z, 0, 1]) * t_inv[mask_z]
            sub_q[mask_z, 1] = (sub_r[mask_z, 2, 0] + sub_r[mask_z, 0, 2]) * t_inv[mask_z]
            sub_q[mask_z, 2] = (sub_r[mask_z, 2, 1] + sub_r[mask_z, 1, 2]) * t_inv[mask_z]
            sub_q[mask_z, 3] = t[mask_z]
            
        q[unsafe_mask] = sub_q

    return NormQuaternion(q)

def _compute_consensus_positions(positions, top_k=3):
    top_k = min(top_k, positions.shape[0])
    mean_pos = positions.mean(dim=0)
    distances = torch.abs(positions - mean_pos).mean(dim=-1)
    topk_val, _ = distances.topk(k=top_k, largest=False)
    threshold = topk_val[-1]
    valid_positions = positions[distances <= threshold]
    return valid_positions.mean(dim=0)

def getTransformation(mob, tar, weights=None, return_rotation=False):

    if weights is None:
        mob_com = mob.mean(0)
        tar_com = tar.mean(0)
        mob = mob - mob_com
        tar = tar - tar_com
        matrix = np.dot(mob.T, tar)
    else:
        weights_sum = weights.sum()
        weights_dot = np.dot(weights.T, weights)
        mob_com = (mob * weights).sum(axis=0) / weights_sum
        tar_com = (tar * weights).sum(axis=0) / weights_sum
        mob = mob - mob_com
        tar = tar - tar_com
        matrix = np.dot((mob * weights).T, (tar * weights)) / weights_dot

    U, _, Vh = np.linalg.svd(matrix)
    d = np.sign(np.linalg.det(np.dot(U, Vh)))
    Id = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, d]])
    rotation = np.dot(Vh.T, np.dot(Id, U.T))
    translation = tar_com - np.dot(mob_com, rotation.T)
    transformed = np.dot(tar, rotation)
    if return_rotation:
        return transformed, rotation
    return transformed


low_prec = True
config = model_config(
    name="model_3",
    train=False, 
    low_prec=low_prec,
)

def getPDB(file):
    CA = []
    with open(file) as fin:
        for line in fin.readlines():
            if line[:4] == 'ATOM':
                if line[13:15] == 'CA':
                    CA.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    CA = torch.tensor(CA)
    dist = CA[None, :, :] - CA[:, None, :]
    dist = (dist * dist).sum(-1).sqrt()
    return dist


seed_num = 42

# model = AlphaFold(config)

# # data = torch.load(f'/export/disk4/wwz/Cerebra/MSA_new_train_checkpoint/model_test_sd_2_{eval_num}.pt', map_location='cpu')
# # data = torch.load(f'/export/disk4/wwz/Cerebra/MSA_new_train_checkpoint/model_test_new_1_{eval_num}.pt', map_location='cpu')
# # data = torch.load(f'/export/disk4/wwz/Cerebra/MSA_new_train_checkpoint/model_test_sd_2_{eval_num}_epoch_test.pt', map_location='cpu')

# data = torch.load(f'/export/disk10/wwz/A800_data/self_distillation_checkpoint/model_test_sd_1_{eval_num}_retrain_fp32.pt', map_location='cpu')
# # data = torch.load(f'/data/2/self_distillation_checkpoint/model_test_sd_2_{eval_num}_new_train.pt', map_location='cpu')
# # elif args.eval_type == 'pfm':
# #     data = torch.load(f'/data/1/PFM_checkpoints/model_PFM_afsd_4_{eval_num}.pt', map_location='cpu')
# model.load_state_dict(data['model_state'])
# ema = True
# if ema:
#     model.load_state_dict(data['ema_state'])
# model = model.to(gpu0)
# model.eval()

# total_params = sum(p.numel() for p in model.parameters())
# print(f'{total_params:,} total parameters.')


np.set_printoptions(precision=4, suppress=True)




# with open('long.txt', 'r') as fin:
#     for i in fin.readlines():
#         tmp = i.strip().split()
#         length =  int(tmp[-1])
#         train_tmp.append([[tmp[0]], length])
# print(len(train_tmp))

restype_3to1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}

restype_1to3 = {}
for k, v in restype_3to1.items():
    restype_1to3[v] = k

AVG_data = []



def load_model(device):
    esm_model_path = "/export/disk1/wwz_backup/esm2_dir/esm2_t36_3B_UR50D.pt"
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet(esm_model_path)
    batch_converter = alphabet.get_batch_converter()
    esm_model.to(device)
    esm_model.eval()  
    model = AlphaFold(config)
    data = torch.load(f'/export/disk10/wwz/A800_data/self_distillation_checkpoint/model_test_sd_1_566_retrain_bf16_psa_li_esm2enhunce_addisorder.pt', map_location='cpu')
 
    model_para =(data['model_state'])
    ema = True
    if ema:
        model_para=(data['ema_state'])
    # model_dict = model.state_dict()
    # # print(set(model_dict.keys()) - set(model_para.keys()))
    # model_para = {k:v for k,v in model_para.items() if k in model_dict.keys()}
    # model_dict.update(model_para)
    model.load_state_dict(model_para)
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    return esm_model,batch_converter,model

def model_package(fasta,esm_model,batch_converter,cerebra_model,device,use_comb=True,use_bf16=False):
    batch = dataloader('pdb',fasta,esm_model, batch_converter,device)
    outputs = run_cycle(batch,cerebra_model,use_bf16=use_bf16)
    length = outputs['translation'][-1].shape[2]
    Anchor = outputs['translation'][-1].shape[1]//2

    all_atom_pos_14, plddt, batch = pdb_file_out(outputs, batch, Anchor, use_comb=use_comb)
    dist = outputs['CE'].reshape(outputs['CE'].shape[0],length,length,36)

    return all_atom_pos_14, plddt,dist,batch

device = torch.device('cuda:7')
path = './tm_FAP1.pdb'
esm_model,batch_converter,model =load_model(device)
fasta = 'DEERLKEILFFLLLIIIFVVFLLIVHYKFLEEFKEKNVTDKEEFNETIKIDEAVMLISAFLLAIAAALLKELEELIRRIPFSEEIRRELEKILGVLAGLYVGAAFLMVIAKFLTKKIKEENTTDKEKLNEWYEFVYVLLIFAFFIILAIAVVLLKLLEILGDEERLKEILFFLLLIIIFVVFLLIVHYKFLEEFKEKNVTDKEEFNETIKIDEAVMLISAFLLAIAAALLKELEELIRRIPFSEEIRRELEKILGVLAGLYVGAAFLMVIAKFLTKKIKEENTTDKEKLNEWYEFVYVLLIFAFFIILAIAVVLLKLLEILG'
all_atom_pos_14, plddt,dist,batch = model_package(fasta,esm_model,batch_converter,model,device,use_bf16=True)
single_pdb_write_IO(all_atom_pos_14,plddt,batch,path)