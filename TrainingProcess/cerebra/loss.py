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

def comp_loss(outputs, batch, AncherList):
    device = outputs['CE'].device
    true_msa = batch['true_msa'].to(device)
    seq = true_msa[:, 0]

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
        'bert_loss': 2.
    }

    loss_items = {}

    loss_items['loss_translation_est'] = torch.stack(outputs['translation_est']).mean()
    loss_items['loss_quaternion_est']  = torch.stack(outputs['quaternion_est']).mean()
    loss_items['loss_CE']              = comp_CE(outputs['CE'], batch['CB_dist'])
    loss_items['loss_Dihedral']        = comp_loss_Dihedral(outputs['PsiPhi'], batch['psi_phi'].to(device))
    loss_items['loss_quaternion'], qAll= comp_loss_quaternion(outputs['quaternion'], batch['quaternion'].to(device))
    loss_items['loss_translation']     = comp_translation(batch['xyz'].to(device), seq, outputs['quaternion'], outputs['translation'], qAll, AncherList)
    loss_items['pLDDT_loss'], loss_items['LDDT_loss'], LDDT, LDDT_loss_best  = comp_pLDDT_loss(batch['xyz'][:, 0, :, 0].to(device), torch.stack(outputs['translation']).permute(1, 0, 2, 3, 4), outputs['pLDDT'], AncherList)
    loss_items['bert_loss']            = comp_bert_Loss(batch['bert_mask'], batch['true_msa'], outputs['bert'])

    ret_lossitems = []

    loss_sum = torch.zeros(1, requires_grad=True).to(device)
    for k, v in loss_items.items():
        ret_lossitems.append(v.detach().cpu().item())
        if torch.isnan(v).any():
            v = torch.zeros(1, requires_grad=True).to(device)
        if torch.isinf(v).any():
            v = torch.zeros(1, requires_grad=True).to(device)
        # if v >= 4:
        #     v = torch.zeros(1, requires_grad=True).to(device)
        # v = torch.clip(v, max=5)
        loss_sum += loss_config[k] * v
        
    length = batch['true_msa'].shape[2]
    loss = loss_sum * np.sqrt(length)

    ret_lossitems.append(loss_sum.detach().cpu().item())
    ret_lossitems.append(loss.detach().cpu().item())
    ret_lossitems.append(LDDT.detach().cpu().item())
    ret_lossitems.append(LDDT_loss_best.detach().cpu().item())
    ret_lossitems = np.array(ret_lossitems)
    return loss, ret_lossitems