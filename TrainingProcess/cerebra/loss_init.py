import torch
import numpy as np
from collections import OrderedDict
import h5py

eps = 1e-6

# def comp_CE(pred, y_dist36bin):
#     pred = pred.double()
#     y_dist36bin = y_dist36bin.to(pred.device)
#     loss_func = torch.nn.CrossEntropyLoss()
#     loss = loss_func(pred.reshape(-1, 36), y_dist36bin.reshape(-1))
#     return loss

def comp_CE(pred, y_dist36bin):
    y_dist36bin = y_dist36bin.to(pred.device).reshape(-1)
    pred = pred.reshape(-1, 36)
    pred = torch.log(torch.softmax(pred + 1e-4, dim=-1) + 1e-4)
    loss_func = torch.nn.NLLLoss()
    loss = loss_func(pred, y_dist36bin)
    return loss

def get_all_atoms(translation, rotation, seq, k):
    def QuaternionMM(q1, q2):
        a = q1[..., 0] * q2[..., 0] - (q1[..., 1:] * q2[..., 1:]).sum(-1)
        bcd = torch.cross(q2[..., 1:], q1[..., 1:], dim=-1) + q1[..., 0].unsqueeze(-1) * q2[..., 1:] + q2[..., 0].unsqueeze(-1) * q1[..., 1:]
        q = torch.cat([a.unsqueeze(-1), bcd], dim=-1)
        return q
        
    def TranslationRotation(q, p):
        p4 = torch.cat([torch.zeros_like(p[..., 0]).unsqueeze(-1), p], dim=-1)
        q_1 = torch.cat([q[..., 0].unsqueeze(-1), -q[..., 1:]], dim=-1)
        return QuaternionMM(QuaternionMM(q, p4), q_1)[..., 1:]

    batch_size, nres = translation.shape[0], translation.shape[2]
    C_N_CB = torch.tensor([
        [1.523, -0.518, -0.537],
        [0.,     1.364, -0.769],
        [0.,     0,     -1.208]
    ]).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(translation.device)

    C  = TranslationRotation(rotation, C_N_CB[..., 0].repeat(batch_size, k, nres, 1)) + translation
    N  = TranslationRotation(rotation, C_N_CB[..., 1].repeat(batch_size, k, nres, 1)) + translation
    CB = TranslationRotation(rotation, C_N_CB[..., 2].repeat(batch_size, k, nres, 1)) + translation
    seq = seq.unsqueeze(1).unsqueeze(-1).to(translation.device)
    CB = torch.where(seq == 5, translation, CB)
    return torch.stack([C, N, CB], dim=-2)

def comp_FAPE(pred, label, cutoff):
    loss_func = torch.nn.MSELoss(reduction='none')
    loss = torch.sqrt(loss_func(pred, label).sum(-1) + eps)
    loss = torch.clamp_max(loss, cutoff)
    return loss.mean()

def comp_FAPE3L(pred, label, cutoff):
    label = label.unsqueeze(3) - label.unsqueeze(2)
    pred = pred.unsqueeze(3) - pred.unsqueeze(2)
    return comp_FAPE(pred, label, cutoff)

def comp_translation(y_CA_Ancher, y_CNCb_Ancher, seq, pred_quaternion, pred_translation, AncherList):
    y_CA_Ancher = y_CA_Ancher[None].repeat(len(pred_quaternion), 1, 1, 1, 1)
    y_CNCb_Ancher = y_CNCb_Ancher[None].repeat(len(pred_quaternion), 1, 1, 1, 1, 1)
    all_C_N_CB = []
    for i in range(len(pred_quaternion)):
        all_C_N_CB.append(get_all_atoms(pred_translation[i], pred_quaternion[i], seq, len(AncherList)))
    all_C_N_CB = torch.stack(all_C_N_CB)
    pred_translation = torch.stack(pred_translation)

    loss_CA_FAPE10 = comp_FAPE(pred_translation, y_CA_Ancher, 10.)
    loss_CA_FAPE_inf = comp_FAPE(pred_translation, y_CA_Ancher, 1000.)
    loss_CNCB_FAPE10 = comp_FAPE(all_C_N_CB, y_CNCb_Ancher, 10.)
    loss_CNCB_FAPE_inf = comp_FAPE(all_C_N_CB, y_CNCb_Ancher, 1000.)

    fape_loss = (loss_CA_FAPE10 * 0.9 + loss_CA_FAPE_inf * 0.1) * 0.05 + (loss_CNCB_FAPE10 * 0.9 + loss_CNCB_FAPE_inf * 0.1) * 0.05
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
    

    pLDDT = pLDDT.squeeze(1)          # [batch, L, L]
    AncherList = torch.LongTensor(AncherList).to(pLDDT.device)

    select_pLDDT = []
    if AncherList.dim() == 1:
        select_pLDDT = pLDDT[:, AncherList]                           # [batch, k/m, L]
    elif AncherList.dim() == 2:
        select_pLDDT = []
        for batch in range(AncherList.shape[0]):
            select_pLDDT.append(pLDDT[batch, AncherList[batch]])
        select_pLDDT = torch.stack(select_pLDDT).transpose(1, 0)         # [batch, k/m, L]

    pLDDT_loss = (select_pLDDT - trueLDDT.detach()[-1]).abs().mean()

    return pLDDT_loss, 1. - LDDT_loss.mean(), LDDT_loss[-1], LDDT_loss_best

def comp_bert_Loss(bert_mask, true_msa, bert):
    bert_mask = bert_mask.to(bert.device)
    true_msa = true_msa.to(bert.device)
    label = true_msa[bert_mask > 0.5].long()
    pred  = bert[bert_mask > 0.5]

    # loss_func = torch.nn.NLLLoss()
    # loss = loss_func(torch.log(torch.softmax(pred + 1e-4, dim=-1) + 1e-4), label)
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(pred, label)

    return loss

def comp_loss_quaternion(pred_quaternion, y_quaternion):
    loss_step = [torch.abs(pred - y_quaternion).mean() for pred in pred_quaternion]
    loss_step = torch.stack(loss_step).mean()
    return loss_step

def comp_loss_Dihedral(pred, y):
    loss = torch.abs(pred - y).mean()
    return loss

def comp_initloss(outputs, batch, AncherList):
    
    device = outputs['CE'].device
    y_CA_Ancher = batch['xyz'][:, AncherList, :, 0].to(device)
    y_CNCb_Ancher = batch['xyz'][:, AncherList, :, 1:].to(device)
    y_quaternion = batch['quaternion'][:, AncherList].to(device)
    true_msa = batch['true_msa'].to(device)
    seq = true_msa[:, 0]

    loss_config = OrderedDict()
    loss_config ={
        'loss_translation_est': 0.0, 
        'loss_quaternion_est': 0.0, 
        'loss_CE': 0.3, 
        'loss_Dihedral': 0.1, 
        'loss_quaternion': 2, 
        'loss_translation': 2, 
        'pLDDT_loss': 0.00, 
        'LDDT_loss': 0.01, 
        'bert_loss': 2.
    }

    loss_items = {}

    loss_items['loss_translation_est'] = torch.stack(outputs['translation_est']).mean()
    loss_items['loss_quaternion_est']  = torch.stack(outputs['quaternion_est']).mean()
    loss_items['loss_CE']              = comp_CE(outputs['CE'], batch['CB_dist'])
    loss_items['loss_Dihedral']        = comp_loss_Dihedral(outputs['PsiPhi'], batch['psi_phi'].to(device))
    loss_items['loss_quaternion']      = comp_loss_quaternion(outputs['quaternion'], y_quaternion)
    loss_items['loss_translation']     = comp_translation(y_CA_Ancher, y_CNCb_Ancher, seq, outputs['quaternion'], outputs['translation'], AncherList)
    loss_items['pLDDT_loss'], loss_items['LDDT_loss'], LDDT, LDDT_loss_best = comp_pLDDT_loss(batch['xyz'][:, 0, :, 0].to(device), torch.stack(outputs['translation']).permute(1, 0, 2, 3, 4), outputs['pLDDT'], AncherList)
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
