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

from openfold.humodel.embedders import InputEmbedder, RecyclingEmbedder
from openfold.humodel.evoformer import EvoformerStack
from openfold.humodel.structure_module import StructureStack

from openfold.utils.tensor_utils import add

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

    def forward(self, feats, prevs, AncherList, _recycle=True):
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

        # [*, N, N, C_z]
        z = add(z, z_prev_emb, inplace=inplace_safe)

        del m_1_prev, z_prev, x_prev, m_1_prev_emb, z_prev_emb   

        m, z, s = self.evoformer(
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
        x1D, x2D, collection_translation, collection_quaternion, collection_translation_est, collection_quaternion_est, pLDDT = self.structure(m, z, AncherList)
        
        
        CE = self.dist36bin(z.permute(0, 3, 1, 2))
        PsiPhi = self.PsiPhi(x1D[:, 0].unsqueeze(1))[..., 0].permute(0, 2, 1)
        
        outputs['translation'] = collection_translation
        outputs['quaternion'] = collection_quaternion
        outputs['translation_est'] = collection_translation_est
        outputs['quaternion_est'] = collection_quaternion_est
        outputs['pLDDT'] = pLDDT
        outputs['CE'] = CE
        outputs['PsiPhi'] = PsiPhi

        x_prev = collection_translation[-1].detach()
        x_prev = x_prev[:, :, None] - x_prev[:, :, :, None]
        x_prev = ((x_prev * x_prev).sum(-1, keepdim=True) + 1e-6).sqrt().mean(1)
        
        # m_1_prev = x1D[..., 0, :, :]
        # z_prev = x2D.permute(0, 2, 3, 1)
        # outputs['bert'] = self.bert(x1D)

        outputs['bert'] = self.bert(m)
        m_1_prev = m[..., 0, :, :]
        z_prev = z
        return m_1_prev, z_prev, x_prev, outputs