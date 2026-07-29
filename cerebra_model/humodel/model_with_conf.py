# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn

from cerebra_model.humodel.embedders import InputEmbedder, RecyclingEmbedder
from cerebra_model.humodel.evoformer import EvoformerStack
from cerebra_model.humodel.structure_module_v2 import StructureStack,AngleResnet
from cerebra_model.humodel.primitives import Linear, LayerNorm
from cerebra_model.humodel.heads import PerResidueLDDTCaPredictor
from cerebra_model.utils.tensor_utils import add

class Dist36bin(nn.Module):
    def __init__(self, h2D_num):
        super(Dist36bin, self).__init__()
        self.Dist = nn.Sequential(
            nn.Conv2d(h2D_num, h2D_num, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(h2D_num, h2D_num, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(h2D_num, 36, 3, padding=1),
        )
                 
    def forward(self, x_2D):
        dist = self.Dist(x_2D)
        dist = (dist.permute(0, 2, 3, 1) + dist.permute(0, 3, 2, 1))/2.
        dist = dist.view(x_2D.shape[0], -1, 36)
        return dist


class AlphaFold(nn.Module):

    def __init__(self, config):
        super(AlphaFold, self).__init__()

        self.globals = config.globals
        self.config = config.model

        # Main trunk + structure module
        self.input_embedder = InputEmbedder(
            **self.config["input_embedder"],
        )
        self.recycling_embedder = RecyclingEmbedder(
            **self.config["recycling_embedder"],
        )

        self.esm2 = Linear(2560, 256, init="relu")
        # Learnable strength for ESM2 fallback when MSA is sparse.
        self.esm2_boost_logit = nn.Parameter(torch.tensor(-4.5))
        self.esm2_boost_max = 1.6
        
        self.evoformer = EvoformerStack(
            **self.config["evoformer_stack"],
        )

        self.structure = StructureStack()

        self.dist36bin = Dist36bin(128)

        self.PsiPhi = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(15, 15), padding=7),
            nn.LeakyReLU(),
            nn.Conv2d(1, 2, kernel_size=(2, 256))
            )
        self.bert = nn.Linear(256, 23)

        # self.m1_gating = nn.Sequential(
        #     Linear(256, 12, init="relu"),
        #     nn.LeakyReLU(),
        #     Linear(12, 1, init="relu")
        # )

        # self.esm2_gating = nn.Sequential(
        #     Linear(256, 12, init="relu"),
        #     nn.LeakyReLU(),
        #     Linear(12, 1, init="relu")
        # )
        self.angle_resnet = AngleResnet(256)
        self.conf_head = Pred_Conf_HeadV2()
    def forward(self, feats, prevs, AncherList, _recycle=True, return_aux=False, return_dist=True, return_conf=True, return_angles=True, reduce_plddt=True, conf_anchor_chunk_size=32, keep_structure_all=False):
        batch_dims = feats["target_feat"].shape[:-2]
        n = feats["target_feat"].shape[-2]
        n_seq = feats["msa_feat"].shape[-3]

        inplace_safe = not (self.training or torch.is_grad_enabled())

        # Prep some features
        seq_mask = feats["seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        msa_mask = feats["msa_mask"]
        
        ## Initialize the MSA and pair representations

        # m: [*, S_c, N, C_m]
        # z: [*, N, N, C_z]
        m, z = self.input_embedder(
            feats["target_feat"],
            feats["residue_index"],
            feats["msa_feat"],
            inplace_safe=inplace_safe,
        )
        # print('0',m,z)
        # torch.save({'m':m,'z':z},'/data2/self_distillation_checkpoint/model_nan_data_detail.pt')
        # print(len(prevs))
        [m_1_prev, z_prev, x_prev] = prevs


        # Initialize the recycling embeddings, if needs be 
        if None in [m_1_prev, z_prev, x_prev]:
            # [*, N, C_m]
            m_1_prev = m.new_zeros(
                (*batch_dims, n, self.config.input_embedder.c_m),
                requires_grad=False,
            )

            # [*, N, N, C_z]
            z_prev = z.new_zeros(
                (*batch_dims, n, n, self.config.input_embedder.c_z),
                requires_grad=False,
            )

            # [*, N, 3]
            x_prev = z.new_zeros(
                (*batch_dims, n, n, 1),
                requires_grad=False,
            )
        x_prev = x_prev.to(dtype=z.dtype)

        # m_1_prev_emb: [*, N, C_m]
        # z_prev_emb: [*, N, N, C_z]
        m_1_prev_emb, z_prev_emb = self.recycling_embedder(
            m_1_prev,
            z_prev,
            x_prev,
            inplace_safe=inplace_safe,
        )

        # [*, S_c, N, C_m]
        m[..., 0, :, :] += m_1_prev_emb
        esm2 = self.esm2(feats['esm2'])[:, None, :, :]
        max_extra_boost = torch.sigmoid(self.esm2_boost_logit) * (
            self.esm2_boost_max - 1.0
        )
        if "msa_depth" in feats:
            msa_depth = feats["msa_depth"].to(dtype=m.dtype, device=m.device)
            if msa_depth.dim() == 0:
                msa_depth = msa_depth.unsqueeze(0)
            if msa_depth.dim() > len(batch_dims):
                reduce_dims = tuple(range(len(batch_dims), msa_depth.dim()))
                msa_depth = msa_depth.mean(dim=reduce_dims)
            msa_insufficiency = (1.0 - msa_depth).clamp(min=0.0, max=1.0)
            msa_insufficiency = msa_insufficiency[(...,) + (None, None, None)]
            esm2_boost = 1.0 + msa_insufficiency * max_extra_boost
        else:
            msa_depth_ratio = msa_mask.to(dtype=m.dtype).mean(dim=-2, keepdim=True)
            msa_insufficiency = 1.0 - msa_depth_ratio
            esm2_boost = 1.0 + msa_insufficiency.unsqueeze(-1) * max_extra_boost
        m = m + esm2 * esm2_boost



        # [*, N, N, C_z]
        z = add(z, z_prev_emb, inplace=inplace_safe)

        del m_1_prev, z_prev, x_prev, m_1_prev_emb, z_prev_emb   

        m, z = self.evoformer(
            m,
            z,
            msa_mask=msa_mask.to(dtype=m.dtype),
            pair_mask=pair_mask.to(dtype=z.dtype),
            chunk_size=self.globals.chunk_size,
            use_lma=self.globals.use_lma,
            use_flash=self.globals.use_flash,
            inplace_safe=inplace_safe,
            _mask_trans=self.config._mask_trans,
        )

        outputs = {}

        x1D, x2D, collection_translation, collection_quaternion, collection_translation_est, collection_quaternion_est = self.structure(m, z, AncherList, keep_all=keep_structure_all)
        
        x_prev = collection_translation[-1].detach()
        x_prev = x_prev[:, :, None] - x_prev[:, :, :, None]
        x_prev = ((x_prev * x_prev).sum(-1, keepdim=True) + 1e-6).sqrt().mean(1)
        
        if return_angles:
            unnormalized_angles, angles = self.angle_resnet(x1D[:,0])
            angles = [unnormalized_angles, angles]
        else:
            angles = None

        if return_conf:
            pLDDT, pAE = self.conf_head(
                x1D[:,0],
                x2D,
                collection_translation[-1],
                collection_quaternion[-1],
                AncherList,
                anchor_chunk_size=conf_anchor_chunk_size,
                reduce_plddt=reduce_plddt,
                return_pae=return_aux,
            )
        else:
            pLDDT, pAE = None, None

        CE = self.dist36bin(z.permute(0, 3, 1, 2)) if return_dist else None
        if return_aux:
            outputs['bert'] = self.bert(m)
        m_1_prev = m[..., 0, :, :]
        z_prev = z
        
        outputs['translation'] = collection_translation
        outputs['quaternion'] = collection_quaternion
        if return_aux:
            outputs['translation_est'] = collection_translation_est
            outputs['quaternion_est'] = collection_quaternion_est
            outputs['PsiPhi'] = self.PsiPhi(x1D[:, 0].unsqueeze(1))[..., 0].permute(0, 2, 1)
            outputs['pAE'] = pAE
        if return_dist:
            outputs['CE'] = CE
        if return_angles:
            outputs['angles'] = angles
        if return_conf:
            outputs['pLDDT'] = pLDDT
        return m_1_prev, z_prev, x_prev, outputs
    
def SelectAncher(embedding,  AncherList, SelectAxis, BatchAxis=None):
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

    
class Pred_Conf_HeadV2(nn.Module):
    def __init__(self):
        super(Pred_Conf_HeadV2, self).__init__()
        self.plddt_head = PerResidueLDDTCaPredictor(50,192,128)
        self.pAE_head = PerResidueLDDTCaPredictor(64,414,128)

    @staticmethod
    def _plddt_from_logits(logits):
        num_bins = logits.shape[-1]
        bin_width = 1.0 / num_bins
        bounds = torch.arange(
            start=0.5 * bin_width,
            end=1.0,
            step=bin_width,
            device=logits.device,
            dtype=logits.dtype,
        )
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return torch.sum(
            probs * bounds.view(*((1,) * len(probs.shape[:-1])), num_bins),
            dim=-1,
        )

    def forward(
        self,
        x1d,
        x2d,
        translation,
        quaternion,
        AncherList,
        anchor_chunk_size=None,
        reduce_plddt=False,
        return_pae=True,
    ):
        AncherList = torch.tensor(AncherList, device=x1d.device)
        num_anchors = AncherList.shape[0]
        pae_output = None

        if return_pae:
            x1d1 = x1d.unsqueeze(1).expand(-1, num_anchors, -1, -1)
            x2d1 = SelectAncher(x2d, AncherList, SelectAxis=2, BatchAxis=0).permute(0, 2, 3, 1) #[b,k,l,c]
            t_dist = (translation**2).sum(-1).sqrt().unsqueeze(-1)   #[b,k,l]
            cut_off = torch.arange(4, 33, 1.0, device=t_dist.device, dtype=t_dist.dtype)
            dist_input = (t_dist <= cut_off[None, None, None, :]).to(dtype=x2d.dtype)
            dist_input = torch.cat((dist_input, (t_dist > 32).to(dtype=x2d.dtype)), dim=-1)
            pae_input = torch.cat((x1d1, x2d1, dist_input), dim=-1)
            pae_output = self.pAE_head(pae_input)

        x2d2 = x2d.permute(0, 2, 3, 1)  #[b,l,l,c]
        x2d2 = (x2d2 + x2d2.transpose(1, 2)).unsqueeze(1)  #[b,1,l,l,c]
        cut_off = torch.arange(1, 33, 0.5, device=x2d.device, dtype=x2d.dtype)

        if not reduce_plddt:
            pred_distmap = ((translation.unsqueeze(-3) - translation.unsqueeze(-2))**2).sum(-1, keepdim=True).sqrt() #[b,k,l,l,1]
            dist_input = (pred_distmap <= cut_off[None, None, None, None, :]).to(dtype=x2d.dtype)
            x2d_full = x2d2.expand(-1, num_anchors, -1, -1, -1)
            pred_lddt_input = torch.cat((x2d_full, dist_input), dim=-1)
            pred_lddt_output = self.plddt_head(pred_lddt_input)  #[b,k,l,l,50]
            return pred_lddt_output, pae_output

        if anchor_chunk_size is None or anchor_chunk_size <= 0:
            anchor_chunk_size = num_anchors
        anchor_chunk_size = max(1, int(anchor_chunk_size))
        pred_lddt_chunks = []
        for start in range(0, num_anchors, anchor_chunk_size):
            end = min(start + anchor_chunk_size, num_anchors)
            translation_chunk = translation[:, start:end]
            pred_distmap = ((translation_chunk.unsqueeze(-3) - translation_chunk.unsqueeze(-2))**2).sum(-1, keepdim=True).sqrt()
            dist_input = (pred_distmap <= cut_off[None, None, None, None, :]).to(dtype=x2d.dtype)
            x2d_chunk = x2d2.expand(-1, end - start, -1, -1, -1)
            pred_lddt_input = torch.cat((x2d_chunk, dist_input), dim=-1)
            pred_lddt_logits = self.plddt_head(pred_lddt_input)
            pred_lddt_chunks.append(self._plddt_from_logits(pred_lddt_logits).mean(dim=-1))
            del translation_chunk, pred_distmap, dist_input, x2d_chunk, pred_lddt_input, pred_lddt_logits

        pred_lddt_output = torch.cat(pred_lddt_chunks, dim=1)  #[b,k,l]
        return pred_lddt_output, pae_output
