import os
import click
import numpy as np
import torch
from data_nomask import *
from openfold.config import model_config
from openfold.humodel.model import AlphaFold
from openfold.np.protein import*
from openfold.utils.tensor_utils import (
    tree_map,
    tensor_tree_map,
    batched_gather,
)
from openfold.humodel.get_all_atoms import  hu_model_pred_to_atom14_pos,make_atom14_masks
from openfold.np import residue_constants as rc

if torch.cuda.is_available() ==True:
    device = torch.device('cuda:0')
else:
     device = torch.device('cpu')

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

def pdb_file_write(pred_output,batch,Ancher_id,path):
    quaternion = pred_output['quaternion'][-1][:,Ancher_id]
    translation = pred_output['translation'][-1][:,Ancher_id]
    angles = pred_output['angles'][-1]
    plddt = pred_output['pLDDT'][:,0,Ancher_id]
    raw_seq = batch['true_msa'][0,0]
    of_seq =[]
    for i in raw_seq:
        of_seq.append(rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE[i])
    of_seq = torch.LongTensor(of_seq).unsqueeze(0)
    all_atom_pos_14 = hu_model_pred_to_atom14_pos(quaternion,translation,angles,of_seq)
    batch['aatype'] = of_seq
    batch = make_atom14_masks(batch)
    single_pdb_write_IO(all_atom_pos_14,plddt,batch,path)

def get_esm(seq):
    gpu = device
    model, alphabet = torch.hub.load('facebookresearch/esm:main', 'esm2_t36_3B_UR50D')
    batch_converter = alphabet.get_batch_converter()
    model.to(gpu)
    model.eval()  # disables dropout for deterministic results

    data = [['arget', seq]]
    with torch.no_grad():
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        results = model(batch_tokens.to(gpu), repr_layers=[36], return_contacts=True)
        
        token_embeds = results["representations"][36] # (batch=1, L+2, dim=2560) token_representations
        token_embeds = token_embeds[:, 1:-1, :].to(dtype=torch.float32).cpu().detach().numpy() # (batch=1, L, dim=2560)

        for idx, v in enumerate(data):
            pdb = v[0]
            
            length = len(v[1])
            return  torch.tensor(token_embeds[idx][:length])
                

def get_feature(msa_file, cycle_num):
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
    with open(msa_file) as fin:
        for line in fin.readlines():
            idx_target += 1
            if idx_target == 2:
                target_seq = line.strip()
            if line[0] != '>':
                seq = []
                for AA in list(line.strip()):
                    if ord(AA) < 97:
                        if AA in HHBLITS_AA_TO_ID.keys():
                            seq.append(HHBLITS_AA_TO_ID[AA])
                        else:
                            seq.append(20)
                msa.append(seq)
    msa = torch.LongTensor(msa)
    esm2 = get_esm(target_seq)

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
    for cycle in range(cycle_num):
        
        
        msa_features = sample_msa(nonensemble_feat.copy(), max_seq, seed_num+cycle)
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
    ret_feature['target_feat'] = torch.stack(target_feat)
    ret_feature['residue_index'] = torch.arange(c1, c2)[None].repeat(cycle_num, 1)
    ret_feature['seq'] = target_seq
    return ret_feature

def dataloader(pdb_list, msa_file):
    mode='eval'
    if mode == 'eval':
        cycle_num = 4
    
    batch_feats = {}
    msa_feat, bert_mask, true_msa = [], [], []
    for pdb in pdb_list[0]:
        # file_dir = f'/home/hujian/data/dataloader/{pdb}.h5'
        
        single_feat = get_feature(msa_file, cycle_num)
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
            
def run_cycle(batch, model):
    m_1_prev, z_prev, x_prev, pLDDT_prev = None, None, None, None
    prevs = [m_1_prev, z_prev, x_prev, pLDDT_prev]

    dims = batch["msa_feat"].shape
    num_iters = dims[0]
    length = dims[3]

    n_clusters = 24

    if 324 >= length >= 224:
        n_clusters = 32
    if length > 324:
        n_clusters = 48
    if length > 500:
        n_clusters = 56

    
    AncherList = np.array([int((x + 1) * length/(n_clusters + 1)) for x in range(n_clusters)])
    AncherList = np.array([int(x* (length - 8)/n_clusters) for x in range(n_clusters)]) + 5
    AncherList = AncherList.astype(int)
    AncherList = np.clip(AncherList, a_min=2, a_max=length-2)

    for cycle_no in range(num_iters):
        is_final_iter = cycle_no == (num_iters - 1)
        feats = {}

        feats["seq_mask"] = torch.ones([dims[1], dims[3]], dtype=torch.float32)
        feats["msa_mask"] = torch.ones([dims[1], dims[2], dims[3]], dtype=torch.float32)

        feats['target_feat'] = batch['target_feat'][cycle_no]
        feats['residue_index'] = batch['residue_index'][cycle_no]

        feats['msa_feat'] = batch['msa_feat'][cycle_no]
        msa_target = feats['msa_feat'][0, 0, :, :23]

        feats['esm2'] = batch['esm2']
        with torch.no_grad():
            dtype = next(model.parameters()).dtype
            device = next(model.parameters()).device
            for k in feats:
                if(feats[k].dtype == torch.float32):
                    feats[k] = feats[k].to(dtype=dtype)
                feats[k] = feats[k].to(device)
            m_1_prev, z_prev, x_prev, pLDDT_prev, outputs = model(feats, prevs, AncherList, _recycle=(num_iters > 1))
            if is_final_iter:
                pass
            else:
                prevs = [m_1_prev.detach(), z_prev.detach(), x_prev.detach(), pLDDT_prev.detach()]
    for k, v in outputs.items():
        if k == 'translation' or k == 'quaternion' or k == 'angles':
            outputs[k] = [x.cpu().detach() for x in v]
    return outputs

def getTransformation(mob, tar, weights=None):

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
    return np.dot(tar, rotation)

working_directory = os.path.abspath(os.path.dirname(__file__))

@click.command()
@click.option('--msa_file', required = True, type = str)
@click.option('--saved_folder', required = True, type = str)
def main(msa_file, saved_folder):
    low_prec = True
    config = model_config(
        name="model_3",
        train=False, 
        low_prec=low_prec,
    )

    batch = dataloader([['']], msa_file)

    for idx in range(1, 6):
        model = AlphaFold(config)
        model.load_state_dict(torch.load(f'{working_directory}/model/model{idx}.pth', map_location='cpu'))
        #model.load_state_dict(torch.load(f'{working_directory}/model/model71.pt', map_location='cpu'))
        model = model.to(device)
        model.eval()
        

        outputs = run_cycle(batch, model)

        xyz = (outputs['translation'][-1][0].cpu()).numpy()
        tmp = []
        for i in range(xyz.shape[0]):
            tmp.append(getTransformation(xyz[8], xyz[i]))
        tmp = torch.tensor(np.array(tmp)).permute(1, 0, 2)
        coors = []
        for t in tmp:
            m = (t - t.mean(0)).abs().mean(-1)
            topk3, _ = m.topk(k= 3)
            topk3 = topk3[-1]
            pos = []
            for i, v in enumerate(m):
                if v < topk3:
                    pos.append(t[i])
            coors.append(torch.stack(pos).mean(0))
        coors = torch.stack(coors)

        outputs['translation'][-1][0][8] = coors
        pdb_file_write(outputs, batch, 8, f'{saved_folder}/test{idx}.pdb')
        print('finish prediction')
if __name__ == '__main__':
    main()
