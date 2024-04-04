import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from cerebra.model.primitives import Linear, LayerNorm
from typing import Tuple
eps = 1e-6

class X1D_UpDate_X2D(nn.Module):
    def __init__(self, dim_x1D, dim_x2D, drouout, head_num):
        super(X1D_UpDate_X2D, self).__init__()
        self.head_num = head_num

        self.scaling = int(dim_x1D/head_num)
        self.Q = nn.Linear(dim_x1D, dim_x1D)
        self.K = nn.Linear(dim_x1D, dim_x1D)
        self.LayerNorm = nn.LayerNorm(dim_x1D)

        self.UpdateX2D = nn.Sequential(
            nn.InstanceNorm2d(head_num * 2 + dim_x2D),
            nn.Conv2d(head_num * 2 + dim_x2D, head_num * 2 + dim_x2D, 3, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm2d(head_num * 2 + dim_x2D),
            nn.Conv2d(head_num * 2 + dim_x2D, head_num * 2 + dim_x2D, 3, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm2d(head_num * 2 + dim_x2D),
            nn.Conv2d(head_num * 2 + dim_x2D, head_num * 2 + dim_x2D, 3, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm2d(head_num * 2 + dim_x2D),
            nn.Conv2d(head_num * 2 + dim_x2D, dim_x2D, 3, padding=1)
        )
    
    def forward(self, x1D, x2D):
        x1D = self.LayerNorm(x1D)
        Q = rearrange(self.Q(x1D), 'b m l (h e) -> b h m l e', h=self.head_num)
        K = rearrange(self.K(x1D), 'b m l (h e) -> b h m l e', h=self.head_num)
        atten1 = torch.einsum('bhmij, bhmkj -> bhmik', [Q, K])/np.sqrt(self.scaling)
        # atten = torch.where(torch.isinf(atten), torch.full_like(atten, 65500),atten)
        # print('X1D_UpDate_X2D:', atten)
        atten = (atten1 + atten1.transpose(3, 4))/2.  # [batch, head, m, L, L]
        atten_MSA = torch.mean(atten, dim=2)
        atten_target = atten[:, :, 0]
        X2D_Updated = self.UpdateX2D(torch.cat((atten_MSA, atten_target, x2D), dim=1))
        return X2D_Updated + x2D

class X2D_UpDate_X1D(nn.Module):
    def __init__(self, dim_x1D, dim_x2D, drouout, head_num):
        super(X2D_UpDate_X1D, self).__init__()
        self.head_num = head_num

        self.AttenMap = nn.Sequential(
            nn.InstanceNorm2d(dim_x2D),
            nn.Conv2d(dim_x2D, dim_x2D, 3, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm2d(dim_x2D),
            nn.Conv2d(dim_x2D, dim_x2D, 3, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm2d(dim_x2D),
            nn.Conv2d(dim_x2D, dim_x2D, 3, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm2d(dim_x2D),
            nn.Conv2d(dim_x2D, head_num, 3, padding=1),
        )
        self.V = nn.Linear(dim_x1D, dim_x1D, bias=False)

        self.fd = nn.Sequential(
            nn.Linear(dim_x1D * head_num, dim_x1D * head_num),
            nn.Dropout(p=0.0),
            nn.LeakyReLU(),
            nn.Linear(dim_x1D * head_num, dim_x1D),
        )
        self.LayerNorm_in = nn.LayerNorm(dim_x1D)
        self.LayerNorm_out = nn.LayerNorm(dim_x1D)
    
    def forward(self, x1D, x2D):
        batch_num, seq_num, seq_length = x1D.shape[0], x1D.shape[1], x1D.shape[2]
        atten = self.AttenMap(x2D)                  # [batch, head, L, L]
        atten = (atten + atten.permute(0, 1, 3, 2))/2.
        atten = torch.softmax(atten+eps, dim=-1).unsqueeze(1)  # [batch, 1, head, L, L]

        x1D = self.LayerNorm_in(x1D)
        X1D_Updated = atten @ (self.V(x1D).unsqueeze(2))                             # [batch, m, head, L, embeding_x1]
        X1D_Updated = X1D_Updated.permute(0, 1, 3, 2, 4).reshape(batch_num, seq_num, seq_length, -1)   # [batch, m, L, embeding_x1*head]
        X1D_Updated = self.LayerNorm_out(self.fd(X1D_Updated) + x1D)
        return X1D_Updated

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv1d, self).__init__()
        self.Conv1d = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding,
            bias=bias
            )
    def forward(self, x, cutoff_list=None):
        if cutoff_list == None or cutoff_list == []:
            return self.Conv1d(x)
        else:
            if max(cutoff_list) - 1 > x.shape[-1] or min(cutoff_list) < 0:
                raise Exception('cutoff_list idx out of range!')
            else:
                cutoff_list_tmp = [0] + cutoff_list + [x.shape[-1]]
                x = torch.cat([self.Conv1d(x[..., cutoff_list_tmp[i]:cutoff_list_tmp[i+1]]) for i in range(len(cutoff_list_tmp) - 1)], dim=-1)
                return x

# net = Conv1d(in_channels=10, out_channels=8, kernel_size=3, stride=2, padding=1)

# x = torch.rand(20, 10, 33)
# print(net(x, [10, 20]).shape)

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d, self).__init__()
        self.Conv2d = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding,
            bias=bias
            )
    def forward(self, x, cutoff_list=None):
        if cutoff_list == None or cutoff_list == []:
            return self.Conv2d(x)
        else:
            if max(cutoff_list) - 1 > x.shape[-1] or min(cutoff_list) < 0:
                raise Exception('cutoff_list idx out of range!')
            else:
                cutoff_list_tmp = [0] + cutoff_list + [x.shape[-1]]
                ret = []
                for i in range(len(cutoff_list_tmp) - 1):
                    ret.append(torch.cat([self.Conv2d(x[..., cutoff_list_tmp[i]:cutoff_list_tmp[i+1], cutoff_list_tmp[j]:cutoff_list_tmp[j+1]]) for j in range(len(cutoff_list_tmp) - 1)], dim=-1))
                return torch.cat(ret, dim=-2)

# net = Conv2d(in_channels=10, out_channels=8, kernel_size=3, stride=1, padding=1)

# x = torch.rand(1, 10, 33, 33)
# print(net(x, []).shape)

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
    
# def TranslationRotation(q, t):
#     if q.dim() == t.dim() and q.dim() > 1 and q.shape[-1] == 4 and t.shape[-1] == 3:
#         t4 = torch.cat([torch.zeros_like(t[..., 0])[..., None], t], dim=-1)
#         q_inv = torch.cat([q[..., 0][..., None], -q[..., 1:]], dim=-1)
#         return QuaternionMM(QuaternionMM(q, t4), q_inv)[..., 1:]
#     else:
#         print('Shape of q&t not correct!')

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

class AttenWeight(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(AttenWeight, self).__init__()
        self.scaling = 1./np.sqrt(dim_out)
        self.LayerNorm = nn.LayerNorm(dim_in)
        self.porj = nn.Linear(dim_in, dim_out)

        self.porj_k = nn.Linear(dim_out, dim_out, bias=False)
        self.porj_q = nn.Linear(dim_out, dim_out, bias=False)

    def forward(self, x):
        x = self.LayerNorm(x)
        x = self.porj(x)                # [head, batch, AncherList, len(AncherList)_Atten, length, dim]
        atten = torch.einsum('h b t i l d, h b t j l d -> h b t i j l', [self.porj_k(x), self.porj_q(x)]) * self.scaling
        atten = torch.softmax(atten+eps, dim=3).mean(4).unsqueeze(-1)
        return atten

def SelectAncher(embedding, AncherList, SelectAxis, BatchAxis=None):
    # embedding: [..., batch, ..., length, ...]
    # AncherList: torch.LongTensor([1, 2, 3]) or torch.LongTensor([[0, 1, 2], [1, 2, 3]])
    # SelectAxis: int
    # BatchAxis:  int
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

class StructureEncoder(nn.Module):
    def __init__(self, dim_x2D, drouout=0.):
        super(StructureEncoder, self).__init__()
        self.Transition = nn.Sequential(
            nn.InstanceNorm2d(dim_x2D + 17*2+30+4),
            Conv2d(dim_x2D + 17*2+30+4, dim_x2D + 17*2+30+4, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(p=drouout),
            nn.InstanceNorm2d(dim_x2D + 17*2+30+4),
            Conv2d(dim_x2D + 17*2+30+4, dim_x2D + 17*2+30+4, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(p=drouout),
            nn.InstanceNorm2d(dim_x2D + 17*2+30+4),
            Conv2d(dim_x2D + 17*2+30+4, dim_x2D, kernel_size=3, padding=1),
        )

    def comp_affinity_map(self, translation):
        # translation [batch, k, length, 3]
        translation_encoding = translation.unsqueeze(2) - translation.unsqueeze(3)                            #[batch, k, length, length, 3]
        translation_encoding = torch.sqrt((translation_encoding * translation_encoding).sum(-1) + 1e-8)       #[batch, k, length, length]
        translation_encoding = translation_encoding.mean(1, keepdim=True)                                     #[batch, 1, length, length]
        translation_encoding_cutoff = (torch.arange(4, 21).float() * 2)[None, :, None, None].to(translation.device)  #[1, 17, 1, 1]
        translation_encoding = torch.sigmoid(translation_encoding - translation_encoding_cutoff)              #[batch, 17, length, length]
        return translation_encoding
    
    def get_atoms_coors(self, quaternion, translation, atoms_type='CB'):
        dim = 2
        if atoms_type == 'C':
            dim = 0
        elif atoms_type == 'N':
            dim = 1
        elif atoms_type == 'CB':
            dim = 2
        else:
            print('Unkonw Atom Type!')

        # quaternion [batch, k, length, 4]
        C_N_CB = torch.tensor([
            [1.523, -0.518, -0.537],
            [0.,     1.364, -0.769],
            [0.,     0,     -1.208]
        ])[None, None, None].to(quaternion.device)
        coors = TranslationRotation(quaternion, C_N_CB[..., dim]) + translation
        return coors

    def translation_K2L(self, translation, quaternion):
        # quaternion [batch, k, length, 4]
        length = quaternion.shape[2]
        quaternion_inv = (quaternion * (torch.tensor([1., -1., -1., -1.]).to(quaternion.device)))[:, :, :, None]
        translation_all = translation[:, :, None] - translation[:, :, :, None]
        translation_L = TranslationRotation(quaternion_inv, translation_all).mean(1)

        cutoff = 1./(torch.arange(start=1, end=11).float() * 10).to(quaternion.device)
        translation_L = rearrange(torch.sigmoid(translation_L[..., None] * cutoff), 'b i j c d -> b (c d) i j')      #[batch, 30, length, length]                                                             
        return translation_L

    def quaternion_K2L(self, quaternion):
        # quaternion [batch, k, length, 4]
        length = quaternion.shape[2]
        quaternion_inv = (quaternion * (torch.tensor([1., -1., -1., -1.]).to(quaternion.device)))[:, :, :, None]
        quaternion_L = NormQuaternionMM(quaternion_inv, quaternion.unsqueeze(2))
        quaternion_L = quaternion_L.mean(1)
        quaternion_L = NormQuaternion(quaternion_L)
        quaternion_L = quaternion_L.permute(0, 3, 1, 2)                                                                #[batch, 4, length, length]
        return quaternion_L
    
    def forward(self, x2D, quaternion, translation):
        # x2D: [batch, dim_in, length, length]
        # translation: [batch, len(AncherList):k, length, 3]
        # quaternion:  [batch, len(AncherList):k, length, 4]

        affinity_mapCA = self.comp_affinity_map(translation)
        affinity_mapCB = self.comp_affinity_map(self.get_atoms_coors(quaternion, translation, atoms_type='CB'))
        translation_L = self.translation_K2L(translation, quaternion)
        quaternion_L = self.quaternion_K2L(quaternion)

        x2D_new = torch.cat([x2D, affinity_mapCB, affinity_mapCA, translation_L, quaternion_L], dim=1)
        
        x2D_new = self.Transition(x2D_new)
        x2D = x2D + x2D_new
        return x2D

class StructureDecoder(nn.Module):
    def __init__(self, dim_x2D, head_num, drouout=0.):
        super(StructureDecoder, self).__init__()
        self.head_num = head_num
        self.proj_pose = nn.Sequential(
            nn.InstanceNorm2d(30+4+dim_x2D),
            Conv2d(30+4+dim_x2D, 30+4+dim_x2D, kernel_size=(1, 7), padding=(0, 3)),
            nn.LeakyReLU(),
            nn.Dropout(p=drouout),
            nn.InstanceNorm2d(30+4+dim_x2D),
            Conv2d(30+4+dim_x2D, 7*head_num, kernel_size=(1, 7), padding=(0, 3))
        )
        
    def forward(self, x2D, quaternion, translation, AncherList):
        # x2D: [batch, dim_x2D, length, length]
        # translation: [batch, len(AncherList):k, length, 3]
        # quaternion:  [batch, len(AncherList):k, length, 4]
        # AncherList: eg [1, 2, 3] or [[1, 2, 3], [3, 4, 5]]

        AncherList = torch.LongTensor(AncherList).to(x2D.device)
        x = SelectAncher(x2D, AncherList, SelectAxis=2, BatchAxis=0).permute(0, 2, 3, 1)   #[batch, len(AncherList):k, length, dim_x2D]

        cutoff = 1./(torch.arange(start=1, end=11).float() * 10).to(translation.device)
        translation_encode = torch.sigmoid(translation[..., None] * cutoff)                #[batch, len(AncherList):k, length, 3, 10]
        translation_encode = rearrange(translation_encode, 'b k l c d -> b k l (c d)')     #[batch, len(AncherList):k, length, 30]

        x = torch.cat([x, translation_encode, quaternion], dim=-1)                         #[batch, len(AncherList):k, length, 30+4+dim_x2D]
        x = x.permute(0, 3, 1, 2)                                                          #[batch, 30+4+dim_x2D, len(AncherList):k, length]
        x = rearrange(self.proj_pose(x), 'b (h d) k l -> h b k l d', d=7)                  #[head_num, batch, len(AncherList):k, length, 7]
        new_translation = x[..., :3].clone()                                               #[head_num, batch, len(AncherList):k, length, 3]
        new_quaternion = NormQuaternion(x[..., 3:].clone())                                #[head_num, batch, len(AncherList):k, length, 4]

        if AncherList.dim() == 1:
            for i in range(AncherList.shape[0]):
                new_quaternion[:, :, i, AncherList[i], 0] = 1.
                new_quaternion[:, :, i, AncherList[i], 1:] = 0.
                new_translation[:, :, i, AncherList[i]] = 0.
        elif AncherList.dim() == 2:
            for batch in range(AncherList.shape[0]):
                for i in range(AncherList.shape[1]):
                    new_quaternion[:, batch, i, AncherList[batch, i], 0] = 1.
                    new_quaternion[:, batch, i, AncherList[batch, i], 1:] = 0.
                    new_translation[:, batch, i, AncherList[batch, i]] = 0.
        # new_translation = translation[None] + new_translation                              #[head_num, batch, len(AncherList):k, length, 3]
        # new_quaternion = NormQuaternionMM(quaternion[None], new_quaternion)              #[head_num, batch, len(AncherList):k, length, 4]
        # Ind_quaternion = torch.tensor([1., 0., 0., 0.]).to(quaternion.device)
        # if (quaternion - Ind_quaternion).abs().sum() > 1e-3:
        #     new_quaternion = NormQuaternion((quaternion[None] + new_quaternion)/2.)
        return new_quaternion, new_translation 

class PathSynthesis(nn.Module):
    def __init__(self, dim_x2D, head_num):
        super(PathSynthesis, self).__init__()
        self.head_num = head_num

        self.weight_quaternion = AttenWeight(int(dim_x2D/self.head_num)*2+4, int(dim_x2D/self.head_num))
        self.weight_translation = AttenWeight(int(dim_x2D/self.head_num)*2+3, int(dim_x2D/self.head_num))

    def GetAttenFeature(self, x2D, AncherList):
        # x2D: [batch, dim_in, length, length]
        # AncherList: eg [1, 2, 3] or [[1, 2, 3], [3, 4, 5]]
        x = SelectAncher(x2D, AncherList, SelectAxis=2, BatchAxis=0).permute(0, 2, 3, 1)
        x = rearrange(x, 'b k l (h d) -> h b k l d', h=self.head_num)
        k, length = x.shape[2:4]

        x_step1_1 = SelectAncher(x, AncherList, SelectAxis=3, BatchAxis=1).unsqueeze(4).repeat(1, 1, 1, 1, length, 1)
        x_step1_2 = x.unsqueeze(2).repeat(1, 1, k, 1, 1, 1)
        x_step1 = torch.cat([x_step1_1, x_step1_2], dim=-1)         #[head, batch, len(AncherList), len(AncherList), length, dim_in*2]
        return x_step1

    def TransformQ(self, quaternion, AncherList, affinity):
        # quaternion: [head, batch, len(AncherList):k, length, 4]
        # AncherList: eg [1, 2, 3] or [[1, 2, 3], [3, 4, 5]]
        head, batch, k, length = quaternion.shape[:4]

        q_step1_1 = SelectAncher(quaternion, AncherList, SelectAxis=3, BatchAxis=1).unsqueeze(4)
        q_step1_2 = quaternion.unsqueeze(2)
        q_step1 = NormQuaternionMM(q_step1_1, q_step1_2)

        q_step1 = q_step1.reshape(head, batch, k, k, length, -1)
        atten = self.weight_quaternion(torch.cat([affinity, q_step1], dim=-1))
        q = NormQuaternion((atten * q_step1).sum(3))

        quaternion_est = q_step1 - NormQuaternion(q.mean(0, keepdim=True)).unsqueeze(3).detach()
        quaternion_est = (quaternion_est * quaternion_est).sum(-1).mean(dim=(0, 3))
        return q, quaternion_est

    def TransformT(self, translation, quaternion, AncherList, affinity):
        # quaternion&translation: [head, batch, len(AncherList), length, 4&3]
        # AncherList: eg [1, 2, 3] or [[1, 2, 3], [3, 4, 5]]
        head, batch, k, length, _ = translation.shape

        q_step1_1 = SelectAncher(quaternion, AncherList, SelectAxis=3, BatchAxis=1).unsqueeze(4)

        t_step1_1 = SelectAncher(translation, AncherList, SelectAxis=3, BatchAxis=1).unsqueeze(4)
        t_step1_2 = translation.unsqueeze(2)
        t_step1 = t_step1_1 + TranslationRotation(q_step1_1, t_step1_2)

        t_step1 = t_step1.reshape(head, batch, k, k, length, -1)

        atten = self.weight_translation(torch.cat([affinity, torch.sigmoid(t_step1)], dim=-1))
        t = (atten * t_step1).sum(3)

        translation_est = t_step1 - t.mean(0, keepdim=True).unsqueeze(3).detach()
        translation_est = (translation_est * translation_est).sum(-1).mean(dim=(0, 3))
        return t, translation_est

    def forward(self, x2D, quaternion, translation, AncherList):
        # quaternion&translation: [head, batch, len(AncherList), length, 4&3]
        # AncherList: eg [1, 2, 3] or [[1, 2, 3], [3, 4, 5]]
        AncherList = torch.LongTensor(AncherList).to(x2D.device)
        affinity = self.GetAttenFeature(x2D, AncherList)

        quaternion, quaternion_est = self.TransformQ(quaternion, AncherList, affinity)
        translation, translation_est = self.TransformT(translation, quaternion, AncherList, affinity)

        quaternion = NormQuaternion(quaternion.mean(dim=0))
        translation = translation.mean(dim=0)

        if AncherList.dim() == 1:
            for i in range(AncherList.shape[0]):
                quaternion[:, i, AncherList[i], 0] = 1.
                quaternion[:, i, AncherList[i], 1:] = 0.
                translation[:, i, AncherList[i]] = 0.
        elif AncherList.dim() == 2:
            for batch in range(AncherList.shape[0]):
                for i in range(AncherList.shape[1]):
                    quaternion[batch, i, AncherList[batch, i], 0] = 1.
                    quaternion[batch, i, AncherList[batch, i], 1:] = 0.
                    translation[batch, i, AncherList[batch, i]] = 0.
        return quaternion, translation, quaternion_est, translation_est

class StructureBlock(nn.Module):
    def __init__(self, dim_x2D, drouout, head_num):
        super(StructureBlock, self).__init__()
        self.StructureEncoder = StructureEncoder(dim_x2D, drouout)
        self.StructureDecoder = StructureDecoder(dim_x2D, head_num, drouout)
        self.PathSynthesis = PathSynthesis(dim_x2D, head_num)

    def forward(self, x2D, quaternion, translation, AncherList):
        x2D = self.StructureEncoder(x2D, quaternion, translation)
        new_quaternion, new_translation = self.StructureDecoder(x2D, quaternion, translation, AncherList)
        new_translation = translation[None] + new_translation
        # new_translation & new_quaternion: [head_num, batch, len(AncherList):k, length, 3/4]

        Ind_quaternion = torch.tensor([1., 0., 0., 0.]).to(quaternion.device)

        batch, k, seq_length = quaternion.shape[:3]
        if seq_length >= 0:
            new_quaternion, new_translation, quaternion_est, translation_est = self.PathSynthesis(x2D, new_quaternion, new_translation, AncherList)
            if (quaternion - Ind_quaternion).abs().sum() > 1e-3:
                new_quaternion = NormQuaternion((quaternion + new_quaternion)/2.)
            return x2D, new_quaternion, new_translation, quaternion_est, translation_est
        else:
            new_translation = new_translation.mean(dim=0)

            new_quaternion = NormQuaternion(new_quaternion.mean(dim=0))
            if (quaternion - Ind_quaternion).abs().sum() > 1e-3:
                new_quaternion = NormQuaternion((quaternion + new_quaternion)/2.)

            est = torch.zeros((batch, k, seq_length)).to(quaternion.device)
            return x2D, new_quaternion, new_translation, est, est


class StructureModule(nn.Module):
    def __init__(self, dim_x1D, dim_x2D, drouout, head_num):
        super(StructureModule, self).__init__()
        # head_num = 1
        self.X1D_UpDate_X2D = X1D_UpDate_X2D(dim_x1D, dim_x2D, drouout, head_num)
        self.StructureBlock = StructureBlock(dim_x2D, drouout, head_num)
        self.X2D_UpDate_X1D = X2D_UpDate_X1D(dim_x1D, dim_x2D, drouout, head_num)

    def forward(self, x1D, x2D, quaternion, translation, AncherList):
        x1D, x2D, quaternion, translation = x1D, x2D, quaternion, translation
        x2D = self.X1D_UpDate_X2D(x1D, x2D)
        x2D, quaternion, translation, quaternion_est, translation_est = self.StructureBlock(x2D, quaternion, translation, AncherList)
        x1D = self.X2D_UpDate_X1D(x1D, x2D)
        return x1D, x2D, quaternion, translation, quaternion_est, translation_est

class StructureStack(nn.Module):
    def __init__(self):
        super(StructureStack, self).__init__()
        dim_x1D, dim_x2D, drouout, head_num = 256, 128, 0., 4
        self.S1 = StructureModule(dim_x1D, dim_x2D, drouout, head_num)
        self.S2 = StructureModule(dim_x1D, dim_x2D, drouout, head_num)
        self.S3 = StructureModule(dim_x1D, dim_x2D, drouout, head_num)
        self.S4 = StructureModule(dim_x1D, dim_x2D, drouout, head_num)
        self.S5 = StructureModule(dim_x1D, dim_x2D, drouout, head_num)
        self.S6 = StructureModule(dim_x1D, dim_x2D, drouout, head_num)
        self.pLDDT = nn.Sequential(
            nn.InstanceNorm2d(dim_x2D),
            Conv2d(dim_x2D, dim_x2D, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(p=drouout),
            nn.InstanceNorm2d(dim_x2D),
            Conv2d(dim_x2D, 1, kernel_size=3, padding=1),
        )
    
    def forward(self, x1D, x2D, AncherList):
        batch, seq_num, length = x1D.shape[:3]
        
        n_clusters = 32
        tmp = torch.LongTensor(AncherList)
        if tmp.dim() == 1:
            n_clusters = tmp.shape[0]
        else:
            n_clusters = tmp.shape[1]
        quaternion = torch.tensor([1., 0., 0., 0.])[None, None, None, :].repeat(batch, n_clusters, length, 1).to(dtype=x1D.dtype)
        translation = torch.tensor([0., 0., 0.])[None, None, None, :].repeat(batch, n_clusters, length, 1).to(dtype=x1D.dtype)
        quaternion = quaternion.to(x1D.device)
        translation = translation.to(x1D.device)
        x2D = x2D.permute(0, 3, 1, 2)
        collection_translation, collection_quaternion = [], []
        collection_translation_est, collection_quaternion_est = [], []

        num = 0
        for s in [self.S1, self.S2, self.S3, self.S4, self.S5, self.S6]:
            x1D, x2D, quaternion, translation, quaternion_est, translation_est = s(x1D, x2D, quaternion, translation, AncherList)
            collection_translation.append(translation)
            collection_quaternion.append(quaternion)
            collection_translation_est.append(torch.sqrt(translation_est + 1e-6).mean())
            collection_quaternion_est.append(torch.sqrt(quaternion_est + 1e-6).mean())

            num += 1
            if num == 6:
                pLDDT = torch.sigmoid(self.pLDDT(x2D.detach()))
        return x1D, x2D, collection_translation, collection_quaternion, collection_translation_est, collection_quaternion_est, pLDDT
    


class AngleResnet(nn.Module):
    """
    Implements Algorithm 20, lines 11-14
    """

    def __init__(self, c_in, c_hidden=128, no_blocks=2, no_angles=7, epsilon = 1e-8):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            no_blocks:
                Number of resnet blocks
            no_angles:
                Number of torsion angles to generate
            epsilon:
                Small constant for normalization
        """
        super(AngleResnet, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.eps = epsilon

        self.linear_in = Linear(self.c_in, self.c_hidden)
        self.linear_initial = Linear(self.c_in, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = AngleResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.linear_out = Linear(self.c_hidden, self.no_angles * 2)

        self.relu = nn.ReLU()

    def forward(
        self, s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, C_hidden] single embedding
        Returns:
            [*, no_angles, 2] predicted angles
        """
        # NOTE: The ReLU's applied to the inputs are absent from the supplement
        # pseudocode but present in the source. For maximal compatibility with
        # the pretrained weights, I'm going with the source.

        # [*, C_hidden]

        s = self.relu(s)
        s = self.linear_in(s)

        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        # [*, no_angles * 2]
        s = self.linear_out(s)

        # [*, no_angles, 2]
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(s ** 2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        s = s / norm_denom

        return unnormalized_s, s
    
class AngleResnetBlock(nn.Module):
    def __init__(self, c_hidden):
        """
        Args:
            c_hidden:
                Hidden channel dimension
        """
        super(AngleResnetBlock, self).__init__()

        self.c_hidden = c_hidden

        self.linear_1 = Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_2 = Linear(self.c_hidden, self.c_hidden, init="final")

        self.relu = nn.ReLU()

    def forward(self, a: torch.Tensor) -> torch.Tensor:

        s_initial = a

        a = self.relu(a)
        a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2(a)

        return a + s_initial