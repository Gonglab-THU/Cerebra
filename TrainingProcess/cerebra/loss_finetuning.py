import torch
import numpy as np
from collections import OrderedDict
import h5py
from cerebra.get_all_atoms import *
from cerebra.np import residue_constants as rc
import ml_collections

eps =1e-7
def SelectAncher(embedding, AncherList, SelectAxis, BatchAxis=None):
    # embedding: [..., batch, ..., length, ...]
    # AncherList: torch.LongTensor([1, 2, 3]) or torch.LongTensor([[0, 1, 2], [1, 2, 3]])
    # SelectAxis: int
    # BatchAxis:  int
    AncherList = torch.LongTensor(AncherList).to(embedding.device)
    if AncherList.dim() == 1:
        ret = embedding.index_select(SelectAxis, AncherList)
        return ret
    elif AncherList.dim() == 2 and BatchAxis != None:
        ret = []
        if SelectAxis == 0:
            if BatchAxis != 0:
                embedding = embedding.transpose(0, BatchAxis)
                for i in range(AncherList.shape[0]):
                    ret.append(embedding[i].index_select(BatchAxis - 1, AncherList[i]))
                ret = torch.stack(ret).transpose(0, BatchAxis)
                return ret
            else:
                print('1 Wrong!!!!')
        else:
            embedding = embedding.transpose(0, BatchAxis)
            for i in range(AncherList.shape[0]):
                ret.append(embedding[i].index_select(SelectAxis - 1, AncherList[i]))
            ret = torch.stack(ret).transpose(0, BatchAxis)
            return ret
    else:
        print('2 Wrong!!!!')

def NormQuaternion(q):
    q = q/torch.sqrt((q * q).sum(-1, keepdim=True) + eps)
    q = torch.sign(torch.sign(q[..., 0]) + 0.5).unsqueeze(-1) * q
    return q

def QuaternionMM(q1, q2):
    if q1.dim() == q2.dim():
        q1_shape = torch.tensor([i for i in q1.shape])
        q2_shape = torch.tensor([i for i in q2.shape])

        q_shape_max, _ = torch.stack([q1_shape, q2_shape]).max(0)

        q1 = q1.repeat(list((q_shape_max/q1_shape).long()))
        q2 = q2.repeat(list((q_shape_max/q2_shape).long()))

        a = q1[..., 0] * q2[..., 0] - (q1[..., 1:] * q2[..., 1:]).sum(-1)
        bcd = torch.cross(q2[..., 1:], q1[..., 1:], dim=-1) + q1[..., 0].unsqueeze(-1) * q2[..., 1:] + q2[..., 0].unsqueeze(-1) * q1[..., 1:]
        q = torch.cat([a.unsqueeze(-1), bcd], dim=-1)
        return q
    else:
        print('Shape of q1&q2 not correct!')

def NormQuaternionMM(q1, q2):
    q = QuaternionMM(q1, q2)
    return NormQuaternion(q)
    

def TranslationRotation(q, t):
    if q.dim() == t.dim() and q.dim() > 1 and q.shape[-1] == 4 and t.shape[-1] == 3:
        q_shape = torch.tensor([i for i in q.shape[:-1]])
        t_shape = torch.tensor([i for i in t.shape[:-1]])
        qt_shape_max, _ = torch.stack([q_shape, t_shape]).max(0)

        q_shape = list(torch.cat([(qt_shape_max/q_shape), torch.ones(1)]).long())
        q = q.repeat(q_shape)

        t_shape = list(torch.cat([(qt_shape_max/t_shape), torch.ones(1)]).long())
        t = t.repeat(t_shape)

        t4 = torch.cat([torch.zeros_like(t[..., 0])[..., None], t], dim=-1)
        q_inv = torch.cat([q[..., 0][..., None], -q[..., 1:]], dim=-1)
        return QuaternionMM(QuaternionMM(q, t4), q_inv)[..., 1:]
    else:
        print('Shape of q&t not correct!')

def comp_CE(pred, y_dist36bin):
    y_dist36bin = y_dist36bin.to(pred.device).reshape(-1)
    pred = pred.reshape(-1, 36)
    pred = torch.log(torch.softmax(pred + 1e-4, dim=-1) + 1e-4)
    loss_func = torch.nn.NLLLoss()
    loss = loss_func(pred, y_dist36bin)
    return loss

def get_all_atoms(translation, rotation, seq):
    k = translation.shape[1]
    batch_size, nres = translation.shape[0], translation.shape[2]
    C_N_CB = torch.tensor([
        [1.523, -0.518, -0.537],
        [0.,     1.364, -0.769],
        [0.,     0,     -1.208]
    ]).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(translation.device)

    C  = TranslationRotation(rotation, C_N_CB[..., 0].repeat(batch_size, k, nres, nres, 1)) + translation
    N  = TranslationRotation(rotation, C_N_CB[..., 1].repeat(batch_size, k, nres, nres, 1)) + translation
    CB = TranslationRotation(rotation, C_N_CB[..., 2].repeat(batch_size, k, nres, nres, 1)) + translation
    seq = seq.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).to(translation.device)
    CB = torch.where(seq == 5, translation, CB)
    return C, N, CB

def comp_translation(y_translation, seq, pred_quaternion_step, pred_translation_step, qAll, AncherList):

    device= pred_quaternion_step[0].device
    
    def comp_FAPE(pred, label):
        loss_func = torch.nn.MSELoss(reduction='none')
        loss = torch.sqrt(loss_func(pred, label).sum(-1) + eps)
        return loss

    k = pred_translation_step[0].shape[1]
    fape_translation = y_translation.unsqueeze(1).repeat(1, k, 1, 1, 1, 1)
    FAPE_loss = []
    for s in range(len(pred_translation_step)):
        pred_quaternion = pred_quaternion_step[s].unsqueeze(-2) * (torch.tensor([1., -1, -1, -1])).to(device)
        pred_CA = TranslationRotation(pred_quaternion, pred_translation_step[s].unsqueeze(-3))
        pred_diag = torch.einsum('bkiid->bkid', pred_CA)
        pred_CA = pred_CA - pred_diag[:, :, :, None]
        pred_C, pred_N, pred_CB = get_all_atoms(pred_CA, qAll[s], seq)
        pred = torch.stack([pred_CA, pred_C, pred_N, pred_CB], dim=-2)
        FAPE_loss.append(comp_FAPE(pred, fape_translation))
    FAPE_loss = torch.stack(FAPE_loss)

    max_cutoff = 10
    FAPE_CA_10 = torch.clamp_max(FAPE_loss[..., 0], max=max_cutoff).mean()/10.
    FAPE_CA_inf = FAPE_loss[..., 0].mean()/10.
    FAPE_CNCB_10 = torch.clamp_max(FAPE_loss[..., 1:], max=max_cutoff).mean()/10.
    FAPE_CNCB_inf = FAPE_loss[..., 1:].mean()/10.

    fape_loss = (FAPE_CA_10 * 0.9 + FAPE_CA_inf * 0.1) * 0.5 + (FAPE_CNCB_10 * 0.9 + FAPE_CNCB_inf * 0.1) * 0.5

    def comp_realFAPE(pred, label):
        label = label.unsqueeze(3) - label.unsqueeze(2)
        pred = pred.unsqueeze(3) - pred.unsqueeze(2)

        loss_func = torch.nn.MSELoss(reduction='none')
        loss = torch.sqrt(loss_func(pred, label).sum(-1) + eps)
        return loss.mean()

    # AncherList = torch.LongTensor(AncherList).to(device)
    # print(AncherList.device, y_translation.device)
    y_CA_Ancher = SelectAncher(y_translation, AncherList, SelectAxis=1, BatchAxis=0)[:, :, :, 0].to(device)
    
    return fape_loss

def comp_pLDDT_loss(y_CA, pred_CA, pLDDT, AncherList):
    def getLDDT(predcadist,truecadist):
        ### predcadist: (N,K,L,L) 由xyzLLL3计算
        ### truecadist: (N,L,L)
        ###
        
        '''
        比较邻居个数
        
        jupyter notebook: /export/disk4/xyg/mixnet/analysis.ipynb

        对于一个残基，考虑序号间隔至少为s的(non local)，并且欧式距离小于15(空间足够接近，存在相互作用)所有残基。
        然后遍历所有残基
        然后计算平均值,如果不取平均值，则可以得到每个残基的lDDT,即一组数据,长度与序列长度同，alphafold2可预测此值
        lDDT分数取值范围：

        D: true distance
        d: predicted distance

        s: minimum sequence separation. lDDT original paper: default s=0
        t: threshold [0.5,1,2,4] the same ones used to compute the GDT-HA score
        R0: inclusion radius,far definition,according to lDDT paper

        Referenece
        0. AlphaFold1 SI
        1. lDDT original paper: doi:10.1093/bioinformatics/btt473
        
        '''
        
        N,K,L,L=predcadist.shape
        truecadist=torch.tile(truecadist[:,None,:,:],(1,K,1,1))
        
        R0=15.0
        maskfar=torch.as_tensor(truecadist<=R0,dtype=torch.float32) # (N,K,L,L)
        
        s=0  #  lDDT original paper: default s=0
        a=torch.arange(L).reshape([1,L]).to(maskfar.device)
        maskLocal=torch.as_tensor(torch.abs(a-a.T)>s,dtype=torch.float32) # (L,L)
        maskLocal=torch.tile(maskLocal[None,None,:,:],(N,K,1,1))
        fenmu=maskLocal*maskfar

        Ratio=0
        t=[0.5,1,2,4] # the same ones used to compute the GDT-HA score
        for t0 in t:
            preserved=torch.as_tensor(torch.abs(truecadist-predcadist)<t0,dtype=torch.float32)
            fenzi=maskLocal*maskfar*preserved
            Ratio+=torch.sum(fenzi,dim=3)/(torch.sum(fenmu,dim=3)+eps)
        lddt=Ratio/4.0  # (N,K,L)  range (0,1]
        return lddt

    def comp_distance(xyz, eps):
        distance = xyz.unsqueeze(-2) - xyz.unsqueeze(-3)
        distance = torch.sqrt((distance * distance).sum(-1) + eps)
        return distance

    # y_CA: [batch, L, 3]   pred_CA: [batch, len(StructureModule), m, L, 3]
    # pLDDT: [len(StructureModule), batch, 1, L, L]
    # pred_CA = pred_CA.detach()
    
    y_ditance = comp_distance(y_CA, eps)              # [batch, L, L]
    pred_distance = comp_distance(pred_CA, eps)       # [batch, len(StructureModule), m, L, L]

    trueLDDT = torch.stack([getLDDT(pred_distance[:, i], y_ditance) for i in range(pred_distance.shape[1])])   # [len(StructureModule), batch, k/m, L]
    LDDT_loss_best = trueLDDT[-1].mean(dim=2).max(dim=1)[0].mean()
    LDDT_loss = trueLDDT.mean(dim=(1, 2, 3))                                                              # [len(StructureModule)]
    LDDT_loss_items = trueLDDT[-1].mean(dim=(1, 2))
    

    pLDDT = pLDDT.squeeze(1)          # [batch, L, L]
    select_pLDDT = SelectAncher(pLDDT, AncherList, SelectAxis=1, BatchAxis=0)
    pLDDT_loss = (select_pLDDT - trueLDDT.detach()[-1]).abs().mean()

    return pLDDT_loss, 1. - LDDT_loss[-1], LDDT_loss[-1], LDDT_loss_best

def comp_bert_Loss(bert_mask, true_msa, bert):
    loss_func_CE = torch.nn.CrossEntropyLoss(reduction='mean')
    bert_mask = bert_mask.to(bert.device)
    true_msa = true_msa.to(bert.device)

    label = true_msa[bert_mask > 0.5].long()
    pred  = bert[bert_mask > 0.5]
    loss = loss_func_CE(pred, label)

    return loss

def comp_loss_quaternion(pred_quaternion, y_quaternion):
    ret_qAll = []
    for i in range(len(pred_quaternion)):
        q_left = pred_quaternion[i] * (torch.tensor([1., -1, -1, -1]).to(y_quaternion.device))
        q_right = pred_quaternion[i].unsqueeze(-3)
        ret_qAll.append(NormQuaternionMM(q_left.unsqueeze(-2), q_right))
    loss = (torch.stack(ret_qAll) - y_quaternion[None, :, None]).abs().mean()
    return loss, ret_qAll

def comp_loss_Dihedral(pred, y):
    loss = torch.abs(pred - y).mean()
    return loss

def side_chain_feats_get(batch,AncherList):
    batch_num = batch['true_msa'].shape[0]
    batch_extra_feat ={}
    for i in range(batch_num):
        if AncherList.dim() == 1:
            single_ancher_list = AncherList
        elif AncherList.dim() == 2:
            single_ancher_list = AncherList[i]

        single_extra_feat = hu_model_single_protein_extra_feat_get(batch['true_msa'][i],batch['all_atoms_pos'][i],batch['all_atoms_mask'][i],single_ancher_list,batch['mask'][i])
        for f in single_extra_feat.keys():
            if f in batch_extra_feat:
                tmp = batch_extra_feat[f]
                tmp.append(single_extra_feat[f])
                batch_extra_feat[f] = tmp
            else:
                tmp = [single_extra_feat[f]]
                batch_extra_feat[f] = tmp
    feats_preload = {}
    feats_preload['residue_index'] = batch['residue_index'][0].unsqueeze(1).repeat(1,len(single_ancher_list),1)

    for k, v in batch_extra_feat.items():
        v = torch.stack(v)
        feats_preload[k] = v
    return feats_preload
def comp_side_chain_loss(outputs, batch, AncherList):
    pred_q = outputs['quaternion'][-1]
    pred_t = outputs['translation'][-1]
    unnorm_angles,angles = outputs['angles']
    aatype = batch['aatype'].to(pred_q.device)
    angles = angles.to(pred_q.device)
    pred_all_atoms_pos = hu_model_pred_to_atom14_pos(pred_q,pred_t,angles,aatype)
    side_chain_fape_loss = comp_single_pdb_sidechain_fape(pred_q,pred_t,pred_all_atoms_pos,batch)
    return side_chain_fape_loss,pred_all_atoms_pos

def chi_angles_loss(outputs,batch):
    unnormalized_angles_sin_cos,angles_sin_cos = outputs['angles']
    aatype = batch['noancher_aatype']
    seq_mask = batch['noancher_seq_mask']
    chi_mask = batch['noancher_torsion_angles_mask'][...,3:]
    chi_angles_sin_cos = batch["noancher_torsion_angles_sin_cos"][..., 3:, :]
    return supervised_chi_loss(angles_sin_cos,unnormalized_angles_sin_cos,aatype,seq_mask,chi_mask,chi_angles_sin_cos)

def comp_finetuningloss(outputs, batch, AncherList,output_pos =False):
    device = outputs['CE'].device
    true_msa = batch['true_msa'].to(device)
    seq = true_msa[:, 0]
    mask = batch['mask'].to(device)

    loss_config = OrderedDict()
    loss_config ={
        'loss_translation_est': 0.01, 
        'loss_quaternion_est': 0.01, 
        'loss_CE': 0.3, 
        'loss_Dihedral': 0.1, 
        'loss_quaternion': 2, 
        'loss_translation': 2, 
        'pLDDT_loss': 0.01, 
        'LDDT_loss': 0.01, 
        'bert_loss': 2,
        'sidechain_fape_loss':0.2,
        'angles_loss':0.01,
        'structure_loss':0.01
    }
    loss_items = {}
    length = batch['true_msa'].shape[2]
    loss_items['loss_translation_est'] = torch.stack(outputs['translation_est']).mean()
    loss_items['loss_quaternion_est']  = torch.stack(outputs['quaternion_est']).mean()

    loss_items['loss_CE']              = comp_CE(outputs['CE'], batch['CB_dist'])
 
    loss_items['loss_Dihedral']        = comp_loss_Dihedral(outputs['PsiPhi'], batch['psi_phi'].to(device))
    
    loss_items['loss_quaternion'], qAll= comp_loss_quaternion(outputs['quaternion'], batch['quaternion'].to(device))
   
    loss_items['loss_translation']     = comp_translation(batch['xyz'].to(device), seq, outputs['quaternion'], outputs['translation'], qAll, AncherList)
 
    loss_items['pLDDT_loss'], loss_items['LDDT_loss'], LDDT, LDDT_loss_best  = comp_pLDDT_loss(batch['xyz'][:, 0, :, 0].to(device), torch.stack(outputs['translation']).permute(1, 0, 2, 3, 4), outputs['pLDDT'], AncherList)
  
    loss_items['bert_loss']            = comp_bert_Loss(batch['bert_mask'], batch['true_msa'], outputs['bert'])

    AncherList = torch.tensor(AncherList,dtype=torch.long,device=batch['xyz'].device)
    batch = side_chain_feats_get(batch,AncherList)
    for k,v in batch.items():
        v= v.to(device)
        batch[k]=v
    AncherList = torch.tensor(AncherList,dtype=torch.long,device=device)
    loss_items['sidechain_fape_loss'],pred_all_atoms_pos = comp_side_chain_loss(outputs, batch, AncherList)
    loss_items['angles_loss'] = chi_angles_loss(outputs,batch)
    loss_items['structure_loss'] = structure_loss(batch,pred_all_atoms_pos)
    ret_lossitems = []

    loss_sum = torch.zeros(1, requires_grad=True).to(device)
    if output_pos ==True:
        loss_sum = torch.zeros(1).to(device)
    for k, v in loss_items.items():
        ret_lossitems.append(v.detach().cpu().item())
        if torch.isnan(v).any():
            v = torch.zeros(1, requires_grad=True).to(device)
        if torch.isinf(v).any():
            v = torch.zeros(1, requires_grad=True).to(device)
        loss_sum += loss_config[k] * v
        
    
    loss = loss_sum * np.sqrt(length)

    ret_lossitems.append(loss_sum.detach().cpu().item())
    ret_lossitems.append(loss.detach().cpu().item())
    ret_lossitems.append(LDDT.detach().cpu().item())
    ret_lossitems.append(LDDT_loss_best.detach().cpu().item())
    ret_lossitems = np.array(ret_lossitems)
    if output_pos ==True:
        return loss, ret_lossitems,pred_all_atoms_pos,batch
    return loss, ret_lossitems





def between_residue_bond_loss(
    pred_atom_positions: torch.Tensor,  # (*, N, 37/14, 3)
    pred_atom_mask: torch.Tensor,  # (*, N, 37/14)
    residue_index: torch.Tensor,  # (*, N)
    aatype: torch.Tensor,  # (*, N)
    tolerance_factor_soft=12.0,
    tolerance_factor_hard=12.0,
    eps=1e-6,
) -> Dict[str, torch.Tensor]:
    """Flat-bottom loss to penalize structural violations between residues.

    This is a loss penalizing any violation of the geometry around the peptide
    bond between consecutive amino acids. This loss corresponds to
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 44, 45.

    Args:
      pred_atom_positions: Atom positions in atom37/14 representation
      pred_atom_mask: Atom mask in atom37/14 representation
      residue_index: Residue index for given amino acid, this is assumed to be
        monotonically increasing.
      aatype: Amino acid type of given residue
      tolerance_factor_soft: soft tolerance factor measured in standard deviations
        of pdb distributions
      tolerance_factor_hard: hard tolerance factor measured in standard deviations
        of pdb distributions

    Returns:
      Dict containing:
        * 'c_n_loss_mean': Loss for peptide bond length violations
        * 'ca_c_n_loss_mean': Loss for violations of bond angle around C spanned
            by CA, C, N
        * 'c_n_ca_loss_mean': Loss for violations of bond angle around N spanned
            by C, N, CA
        * 'per_residue_loss_sum': sum of all losses for each residue
        * 'per_residue_violation_mask': mask denoting all residues with violation
            present.
    """
    # Get the positions of the relevant backbone atoms.
    this_ca_pos = pred_atom_positions[..., :-1, 1, :]
    this_ca_mask = pred_atom_mask[..., :-1, 1]
    this_c_pos = pred_atom_positions[..., :-1, 2, :]
    this_c_mask = pred_atom_mask[..., :-1, 2]
    next_n_pos = pred_atom_positions[..., 1:, 0, :]
    next_n_mask = pred_atom_mask[..., 1:, 0]
    next_ca_pos = pred_atom_positions[..., 1:, 1, :]
    next_ca_mask = pred_atom_mask[..., 1:, 1]
    has_no_gap_mask = (residue_index[..., 1:] - residue_index[..., :-1]) == 1.0

    # Compute loss for the C--N bond.
    c_n_bond_length = torch.sqrt(
        eps + torch.sum((this_c_pos - next_n_pos) ** 2, dim=-1)
    )

    # The C-N bond to proline has slightly different length because of the ring.
    next_is_proline = aatype[..., 1:] == rc.resname_to_idx["PRO"]
    gt_length = (
        ~next_is_proline
    ) * rc.between_res_bond_length_c_n[
        0
    ] + next_is_proline * rc.between_res_bond_length_c_n[
        1
    ]
    gt_stddev = (
        ~next_is_proline
    ) * rc.between_res_bond_length_stddev_c_n[
        0
    ] + next_is_proline * rc.between_res_bond_length_stddev_c_n[
        1
    ]
    c_n_bond_length_error = torch.sqrt(eps + (c_n_bond_length - gt_length) ** 2)
    c_n_loss_per_residue = torch.nn.functional.relu(
        c_n_bond_length_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_c_mask * next_n_mask * has_no_gap_mask
    c_n_loss = torch.sum(mask * c_n_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    c_n_violation_mask = mask * (
        c_n_bond_length_error > (tolerance_factor_hard * gt_stddev)
    )

    # Compute loss for the angles.
    ca_c_bond_length = torch.sqrt(
        eps + torch.sum((this_ca_pos - this_c_pos) ** 2, dim=-1)
    )
    n_ca_bond_length = torch.sqrt(
        eps + torch.sum((next_n_pos - next_ca_pos) ** 2, dim=-1)
    )

    c_ca_unit_vec = (this_ca_pos - this_c_pos) / ca_c_bond_length[..., None]
    c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length[..., None]
    n_ca_unit_vec = (next_ca_pos - next_n_pos) / n_ca_bond_length[..., None]

    ca_c_n_cos_angle = torch.sum(c_ca_unit_vec * c_n_unit_vec, dim=-1)
    gt_angle = rc.between_res_cos_angles_ca_c_n[0]
    gt_stddev = rc.between_res_bond_length_stddev_c_n[0]
    ca_c_n_cos_angle_error = torch.sqrt(
        eps + (ca_c_n_cos_angle - gt_angle) ** 2
    )
    ca_c_n_loss_per_residue = torch.nn.functional.relu(
        ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
    ca_c_n_loss = torch.sum(mask * ca_c_n_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    ca_c_n_violation_mask = mask * (
        ca_c_n_cos_angle_error > (tolerance_factor_hard * gt_stddev)
    )

    c_n_ca_cos_angle = torch.sum((-c_n_unit_vec) * n_ca_unit_vec, dim=-1)
    gt_angle = rc.between_res_cos_angles_c_n_ca[0]
    gt_stddev = rc.between_res_cos_angles_c_n_ca[1]
    c_n_ca_cos_angle_error = torch.sqrt(
        eps + torch.square(c_n_ca_cos_angle - gt_angle)
    )
    c_n_ca_loss_per_residue = torch.nn.functional.relu(
        c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask
    c_n_ca_loss = torch.sum(mask * c_n_ca_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    c_n_ca_violation_mask = mask * (
        c_n_ca_cos_angle_error > (tolerance_factor_hard * gt_stddev)
    )

    # Compute a per residue loss (equally distribute the loss to both
    # neighbouring residues).
    per_residue_loss_sum = (
        c_n_loss_per_residue + ca_c_n_loss_per_residue + c_n_ca_loss_per_residue
    )
    per_residue_loss_sum = 0.5 * (
        torch.nn.functional.pad(per_residue_loss_sum, (0, 1))
        + torch.nn.functional.pad(per_residue_loss_sum, (1, 0))
    )

    # Compute hard violations.
    violation_mask = torch.max(
        torch.stack(
            [c_n_violation_mask, ca_c_n_violation_mask, c_n_ca_violation_mask],
            dim=-2,
        ),
        dim=-2,
    )[0]
    violation_mask = torch.maximum(
        torch.nn.functional.pad(violation_mask, (0, 1)),
        torch.nn.functional.pad(violation_mask, (1, 0)),
    )

    return {
        "c_n_loss_mean": c_n_loss,
        "ca_c_n_loss_mean": ca_c_n_loss,
        "c_n_ca_loss_mean": c_n_ca_loss,
        "per_residue_loss_sum": per_residue_loss_sum,
        "per_residue_violation_mask": violation_mask,
    }

def between_residue_clash_loss(
    atom14_pred_positions: torch.Tensor,
    atom14_atom_exists: torch.Tensor,
    atom14_atom_radius: torch.Tensor,
    residue_index: torch.Tensor,
    overlap_tolerance_soft=1.5,
    overlap_tolerance_hard=1.5,
    eps=1e-10,
) -> Dict[str, torch.Tensor]:
    """Loss to penalize steric clashes between residues.

    This is a loss penalizing any steric clashes due to non bonded atoms in
    different peptides coming too close. This loss corresponds to the part with
    different residues of
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

    Args:
      atom14_pred_positions: Predicted positions of atoms in
        global prediction frame
      atom14_atom_exists: Mask denoting whether atom at positions exists for given
        amino acid type
      atom14_atom_radius: Van der Waals radius for each atom.
      residue_index: Residue index for given amino acid.
      overlap_tolerance_soft: Soft tolerance factor.
      overlap_tolerance_hard: Hard tolerance factor.

    Returns:
      Dict containing:
        * 'mean_loss': average clash loss
        * 'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 14)
        * 'per_atom_clash_mask': mask whether atom clashes with any other atom
            shape (N, 14)
    """
    fp_type = atom14_pred_positions.dtype

    # Create the distance matrix.
    # (N, N, 14, 14)
    dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_pred_positions[..., :, None, :, None, :]
                - atom14_pred_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    # Create the mask for valid distances.
    # shape (N, N, 14, 14)
    dists_mask = (
        atom14_atom_exists[..., :, None, :, None]
        * atom14_atom_exists[..., None, :, None, :]
    ).type(fp_type)

    # Mask out all the duplicate entries in the lower triangular matrix.
    # Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
    # are handled separately.
    dists_mask = dists_mask * (
        residue_index[..., :, None, None, None]
        < residue_index[..., None, :, None, None]
    )

    # Backbone C--N bond between subsequent residues is no clash.
    c_one_hot = torch.nn.functional.one_hot(
        residue_index.new_tensor(2), num_classes=14
    )
    c_one_hot = c_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *c_one_hot.shape
    )
    c_one_hot = c_one_hot.type(fp_type)
    n_one_hot = torch.nn.functional.one_hot(
        residue_index.new_tensor(0), num_classes=14
    )
    n_one_hot = n_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *n_one_hot.shape
    )
    n_one_hot = n_one_hot.type(fp_type)

    neighbour_mask = (
        residue_index[..., :, None, None, None] + 1
    ) == residue_index[..., None, :, None, None]
    c_n_bonds = (
        neighbour_mask
        * c_one_hot[..., None, None, :, None]
        * n_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - c_n_bonds)

    # Disulfide bridge between two cysteines is no clash.
    cys = rc.restype_name_to_atom14_names["CYS"]
    cys_sg_idx = cys.index("SG")
    cys_sg_idx = residue_index.new_tensor(cys_sg_idx)
    cys_sg_idx = cys_sg_idx.reshape(
        *((1,) * len(residue_index.shape[:-1])), 1
    ).squeeze(-1)
    cys_sg_one_hot = torch.nn.functional.one_hot(cys_sg_idx, num_classes=14)
    disulfide_bonds = (
        cys_sg_one_hot[..., None, None, :, None]
        * cys_sg_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - disulfide_bonds)

    # Compute the lower bound for the allowed distances.
    # shape (N, N, 14, 14)
    dists_lower_bound = dists_mask * (
        atom14_atom_radius[..., :, None, :, None]
        + atom14_atom_radius[..., None, :, None, :]
    )

    # Compute the error.
    # shape (N, N, 14, 14)
    dists_to_low_error = dists_mask * torch.nn.functional.relu(
        dists_lower_bound - overlap_tolerance_soft - dists
    )

    # Compute the mean loss.
    # shape ()
    mean_loss = torch.sum(dists_to_low_error) / (1e-6 + torch.sum(dists_mask))

    # Compute the per atom loss sum.
    # shape (N, 14)
    per_atom_loss_sum = torch.sum(dists_to_low_error, dim=(-4, -2)) + torch.sum(
        dists_to_low_error, axis=(-3, -1)
    )

    # Compute the hard clash mask.
    # shape (N, N, 14, 14)
    clash_mask = dists_mask * (
        dists < (dists_lower_bound - overlap_tolerance_hard)
    )

    # Compute the per atom clash.
    # shape (N, 14)
    per_atom_clash_mask = torch.maximum(
        torch.amax(clash_mask, axis=(-4, -2)),
        torch.amax(clash_mask, axis=(-3, -1)),
    )

    return {
        "mean_loss": mean_loss,  # shape ()
        "per_atom_loss_sum": per_atom_loss_sum,  # shape (N, 14)
        "per_atom_clash_mask": per_atom_clash_mask,  # shape (N, 14)
    }

def within_residue_violations(
    atom14_pred_positions: torch.Tensor,
    atom14_atom_exists: torch.Tensor,
    atom14_dists_lower_bound: torch.Tensor,
    atom14_dists_upper_bound: torch.Tensor,
    tighten_bounds_for_loss=0.0,
    eps=1e-10,
) -> Dict[str, torch.Tensor]:

    dists_masks = 1.0 - torch.eye(14, device=atom14_atom_exists.device)[None]
    dists_masks = dists_masks.reshape(
        *((1,) * len(atom14_atom_exists.shape[:-2])), *dists_masks.shape
    )
    dists_masks = (
        atom14_atom_exists[..., :, :, None]
        * atom14_atom_exists[..., :, None, :]
        * dists_masks
    )

    # Distance matrix
    dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_pred_positions[..., :, :, None, :]
                - atom14_pred_positions[..., :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    # Compute the loss.
    dists_to_low_error = torch.nn.functional.relu(
        atom14_dists_lower_bound + tighten_bounds_for_loss - dists
    )
    dists_to_high_error = torch.nn.functional.relu(
        dists - (atom14_dists_upper_bound - tighten_bounds_for_loss)
    )
    loss = dists_masks * (dists_to_low_error + dists_to_high_error)

    # Compute the per atom loss sum.
    per_atom_loss_sum = torch.sum(loss, dim=-2) + torch.sum(loss, dim=-1)

    # Compute the violations mask.
    violations = dists_masks * (
        (dists < atom14_dists_lower_bound) | (dists > atom14_dists_upper_bound)
    )

    # Compute the per atom violations.
    per_atom_violations = torch.maximum(
        torch.max(violations, dim=-2)[0], torch.max(violations, axis=-1)[0]
    )

    return {
        "per_atom_loss_sum": per_atom_loss_sum,
        "per_atom_violations": per_atom_violations,
    }


def find_structural_violations(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,
    violation_tolerance_factor: float,
    clash_overlap_tolerance: float,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Computes several checks for structural violations."""

    # Compute between residue backbone violations of bonds and angles.
    connection_violations = between_residue_bond_loss(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=batch["atom14_atom_exists"],
        residue_index=batch["residue_index"],
        aatype=batch["aatype"],
        tolerance_factor_soft=violation_tolerance_factor,
        tolerance_factor_hard=violation_tolerance_factor,
    )

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # Shape: (N, 14).
    atomtype_radius = [
        rc.van_der_waals_radius[name[0]]
        for name in rc.atom_types
    ]
    atomtype_radius = atom14_pred_positions.new_tensor(atomtype_radius)
    atom14_atom_radius = (
        batch["atom14_atom_exists"]
        * atomtype_radius[batch["residx_atom14_to_atom37"]]
    )

    # Compute the between residue clash loss.
    between_residue_clashes = between_residue_clash_loss(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=batch["atom14_atom_exists"],
        atom14_atom_radius=atom14_atom_radius,
        residue_index=batch["residue_index"],
        overlap_tolerance_soft=clash_overlap_tolerance,
        overlap_tolerance_hard=clash_overlap_tolerance,
    )

    # Compute all within-residue violations (clashes,
    # bond length and angle violations).
    restype_atom14_bounds = rc.make_atom14_dists_bounds(
        overlap_tolerance=clash_overlap_tolerance,
        bond_length_tolerance_factor=violation_tolerance_factor,
    )
    atom14_atom_exists = batch["atom14_atom_exists"]
    atom14_dists_lower_bound = atom14_pred_positions.new_tensor(
        restype_atom14_bounds["lower_bound"]
    )[batch["aatype"]]
    atom14_dists_upper_bound = atom14_pred_positions.new_tensor(
        restype_atom14_bounds["upper_bound"]
    )[batch["aatype"]]
    residue_violations = within_residue_violations(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=batch["atom14_atom_exists"],
        atom14_dists_lower_bound=atom14_dists_lower_bound,
        atom14_dists_upper_bound=atom14_dists_upper_bound,
        tighten_bounds_for_loss=0.0,
    )

    # Combine them to a single per-residue violation mask (used later for LDDT).
    per_residue_violations_mask = torch.max(
        torch.stack(
            [
                connection_violations["per_residue_violation_mask"],
                torch.max(
                    between_residue_clashes["per_atom_clash_mask"], dim=-1
                )[0],
                torch.max(residue_violations["per_atom_violations"], dim=-1)[0],
            ],
            dim=-1,
        ),
        dim=-1,
    )[0]

    return {
        "between_residues": {
            "bonds_c_n_loss_mean": connection_violations["c_n_loss_mean"],  # ()
            "angles_ca_c_n_loss_mean": connection_violations[
                "ca_c_n_loss_mean"
            ],  # ()
            "angles_c_n_ca_loss_mean": connection_violations[
                "c_n_ca_loss_mean"
            ],  # ()
            "connections_per_residue_loss_sum": connection_violations[
                "per_residue_loss_sum"
            ],  # (N)
            "connections_per_residue_violation_mask": connection_violations[
                "per_residue_violation_mask"
            ],  # (N)
            "clashes_mean_loss": between_residue_clashes["mean_loss"],  # ()
            "clashes_per_atom_loss_sum": between_residue_clashes[
                "per_atom_loss_sum"
            ],  # (N, 14)
            "clashes_per_atom_clash_mask": between_residue_clashes[
                "per_atom_clash_mask"
            ],  # (N, 14)
        },
        "within_residues": {
            "per_atom_loss_sum": residue_violations[
                "per_atom_loss_sum"
            ],  # (N, 14)
            "per_atom_violations": residue_violations[
                "per_atom_violations"
            ],  # (N, 14),
        },
        "total_per_residue_violations_mask": per_residue_violations_mask,  # (N)
    }





def extreme_ca_ca_distance_violations(
    pred_atom_positions: torch.Tensor,  # (N, 37(14), 3)
    pred_atom_mask: torch.Tensor,  # (N, 37(14))
    residue_index: torch.Tensor,  # (N)
    max_angstrom_tolerance=1.5,
    eps=1e-6,
) -> torch.Tensor:
    """Counts residues whose Ca is a large distance from its neighbour.

    Measures the fraction of CA-CA pairs between consecutive amino acids that are
    more than 'max_angstrom_tolerance' apart.

    Args:
      pred_atom_positions: Atom positions in atom37/14 representation
      pred_atom_mask: Atom mask in atom37/14 representation
      residue_index: Residue index for given amino acid, this is assumed to be
        monotonically increasing.
      max_angstrom_tolerance: Maximum distance allowed to not count as violation.
    Returns:
      Fraction of consecutive CA-CA pairs with violation.
    """
    this_ca_pos = pred_atom_positions[..., :-1, 1, :]
    this_ca_mask = pred_atom_mask[..., :-1, 1]
    next_ca_pos = pred_atom_positions[..., 1:, 1, :]
    next_ca_mask = pred_atom_mask[..., 1:, 1]
    has_no_gap_mask = (residue_index[..., 1:] - residue_index[..., :-1]) == 1.0
    ca_ca_distance = torch.sqrt(
        eps + torch.sum((this_ca_pos - next_ca_pos) ** 2, dim=-1)
    )
    violations = (
        ca_ca_distance - rc.ca_ca
    ) > max_angstrom_tolerance
    mask = this_ca_mask * next_ca_mask * has_no_gap_mask
    mean = masked_mean(mask, violations, -1)
    return mean


def compute_violation_metrics(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,  # (N, 14, 3)
    violations: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute several metrics to assess the structural violations."""
    ret = {}
    extreme_ca_ca_violations = extreme_ca_ca_distance_violations(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=batch["atom14_atom_exists"],
        residue_index=batch["residue_index"],
    )
    ret["violations_extreme_ca_ca_distance"] = extreme_ca_ca_violations
    ret["violations_between_residue_bond"] = masked_mean(
        batch["seq_mask"],
        violations["between_residues"][
            "connections_per_residue_violation_mask"
        ],
        dim=-1,
    )
    ret["violations_between_residue_clash"] = masked_mean(
        mask=batch["seq_mask"],
        value=torch.max(
            violations["between_residues"]["clashes_per_atom_clash_mask"],
            dim=-1,
        )[0],
        dim=-1,
    )
    ret["violations_within_residue"] = masked_mean(
        mask=batch["seq_mask"],
        value=torch.max(
            violations["within_residues"]["per_atom_violations"], dim=-1
        )[0],
        dim=-1,
    )
    ret["violations_per_residue"] = masked_mean(
        mask=batch["seq_mask"],
        value=violations["total_per_residue_violations_mask"],
        dim=-1,
    )
    return ret


def violation_loss(
    violations: Dict[str, torch.Tensor],
    atom14_atom_exists: torch.Tensor,
    eps=1e-6,
    **kwargs,
) -> torch.Tensor:
    num_atoms = torch.sum(atom14_atom_exists)
    l_clash = torch.sum(
        violations["between_residues"]["clashes_per_atom_loss_sum"]
        + violations["within_residues"]["per_atom_loss_sum"]
    )
    l_clash = l_clash / (eps + num_atoms)
    loss = (
        violations["between_residues"]["bonds_c_n_loss_mean"]
        + violations["between_residues"]["angles_ca_c_n_loss_mean"]
        + violations["between_residues"]["angles_c_n_ca_loss_mean"]
        + l_clash
    )
    mean = torch.mean(loss)

    return mean

def structure_loss(batch,atom_14_positions):
    violation_config= {
                "violation_tolerance_factor": 12.0,
                "clash_overlap_tolerance": 1.5,
                "eps": eps,  # 1e-6,
                "weight": 0.0,
            }
    violation =  find_structural_violations(batch,atom_14_positions,**violation_config)
    comp_violation_loss = violation_loss(violation,batch["atom14_atom_exists"])
    return comp_violation_loss


def find_structural_violations_np(
    batch: Dict[str, np.ndarray],
    atom14_pred_positions: np.ndarray,
    config: ml_collections.ConfigDict,
) -> Dict[str, np.ndarray]:
    to_tensor = lambda x: torch.tensor(x)
    batch = tree_map(to_tensor, batch, np.ndarray)
    atom14_pred_positions = to_tensor(atom14_pred_positions)

    out = find_structural_violations(batch, atom14_pred_positions, **config)

    to_np = lambda x: np.array(x)
    np_out = tensor_tree_map(to_np, out)

    return np_out

def compute_violation_metrics(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,  # (N, 14, 3)
    violations: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute several metrics to assess the structural violations."""
    ret = {}
    extreme_ca_ca_violations = extreme_ca_ca_distance_violations(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=batch["atom14_atom_exists"],
        residue_index=batch["residue_index"],
    )
    ret["violations_extreme_ca_ca_distance"] = extreme_ca_ca_violations
    ret["violations_between_residue_bond"] = masked_mean(
        batch["seq_mask"],
        violations["between_residues"][
            "connections_per_residue_violation_mask"
        ],
        dim=-1,
    )
    ret["violations_between_residue_clash"] = masked_mean(
        mask=batch["seq_mask"],
        value=torch.max(
            violations["between_residues"]["clashes_per_atom_clash_mask"],
            dim=-1,
        )[0],
        dim=-1,
    )
    ret["violations_within_residue"] = masked_mean(
        mask=batch["seq_mask"],
        value=torch.max(
            violations["within_residues"]["per_atom_violations"], dim=-1
        )[0],
        dim=-1,
    )
    ret["violations_per_residue"] = masked_mean(
        mask=batch["seq_mask"],
        value=violations["total_per_residue_violations_mask"],
        dim=-1,
    )
    return ret


def compute_violation_metrics_np(
    batch: Dict[str, np.ndarray],
    atom14_pred_positions: np.ndarray,
    violations: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    to_tensor = lambda x: torch.tensor(x)
    batch = tree_map(to_tensor, batch, np.ndarray)
    atom14_pred_positions = to_tensor(atom14_pred_positions)
    violations = tree_map(to_tensor, violations, np.ndarray)

    out = compute_violation_metrics(batch, atom14_pred_positions, violations)

    to_np = lambda x: np.array(x)
    return tree_map(to_np, out, torch.Tensor)
