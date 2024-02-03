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

import os, random
import numpy as np
import torch
from typing import Mapping, Dict
from functools import reduce
from operator import add
from random import shuffle


eps = 1e-7
def setup_seed(seed):
    os.environ['PYTHONSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_num = 42

setup_seed(seed_num)
FeatureDict = Mapping[str, np.ndarray]

def make_msa_features(msa) -> FeatureDict:
    unique_msa, inverse_indices = torch.unique(msa, sorted=False, return_inverse=True, dim=0)
    t1 = unique_msa[0].clone()
    unique_msa[0] = msa[0]
    unique_msa[inverse_indices[0]] = t1

    features = {}
    # features["msa"] = unique_msa[:2000]
    features["msa"] = unique_msa
    features["deletion_matrix_int"] = torch.zeros_like(unique_msa)
    features["deletion_matrix"] = torch.zeros_like(unique_msa)
    return features

def make_msa_mask(protein):
    """Mask features are all ones, but will later be zero-padded."""
    protein["msa_mask"] = torch.ones(protein["msa"].shape, dtype=torch.float32)
    return protein

MSA_FEATURE_NAMES = [
    "msa",
    "deletion_matrix",
    "msa_mask",
    "bert_mask",
    "true_msa",
]

def make_one_hot(x, num_classes):
    x = x.type(torch.int64)
    x_one_hot = torch.zeros(*x.shape, num_classes, device=x.device)
    x_one_hot.scatter_(-1, x.unsqueeze(-1).long(), 1)
    return x_one_hot

def make_hhblits_profile(protein):
    """Compute the HHblits MSA profile if not already present."""
    if "hhblits_profile" in protein:
        return protein
    # Compute the profile for every residue (over all MSA sequences).
    msa_one_hot = make_one_hot(protein["msa"], 22)

    protein["hhblits_profile"] = torch.mean(msa_one_hot, dim=0)
    return protein

def sample_msa(protein, max_seq, seed, keep_extra=True):
    """Sample MSA randomly, remaining sequences are stored are stored as `extra_*`.""" 
    num_seq = protein["msa"].shape[0]
    g = torch.Generator(device=protein["msa"].device)
    if seed is not None:
        g.manual_seed(seed)
    else:
        g.seed()
    shuffled = torch.randperm(num_seq - 1, generator=g) + 1
    index_order = torch.cat((torch.tensor([0], device=shuffled.device), shuffled), dim=0)
    num_sel = min(max_seq, num_seq)
    sel_seq, not_sel_seq = torch.split(index_order, [num_sel, num_seq - num_sel])
    for k in MSA_FEATURE_NAMES:
        if k in protein:
            if keep_extra:
                protein["extra_" + k] = torch.index_select(protein[k], 0, not_sel_seq)
            protein[k] = torch.index_select(protein[k], 0, sel_seq)
    return protein

def shaped_categorical(probs, epsilon=1e-10):
    ds = probs.shape
    num_classes = ds[-1]
    distribution = torch.distributions.categorical.Categorical(
        torch.reshape(probs + epsilon, [-1, num_classes])
    )
    counts = distribution.sample()
    return torch.reshape(counts, ds[:-1])

def make_masked_msa(protein, mode, replace_fraction=0.15):
    profile_prob = same_prob = uniform_prob = 0.1
    if mode == 'eval':
        replace_fraction=0.15
    """Create data for BERT on raw MSA."""
    # Add a random amino acid uniformly.
    random_aa = torch.tensor([0.05] * 20 + [0.0, 0.0], dtype=torch.float32, device=protein["msa"].device)

    categorical_probs = (uniform_prob * random_aa+ profile_prob * protein["hhblits_profile"]+ same_prob * make_one_hot(protein["msa"], 22))

    # Put all remaining probability on [MASK] which is a new column
    pad_shapes = list(reduce(add, [(0, 0) for _ in range(len(categorical_probs.shape))]))
    pad_shapes[1] = 1
    mask_prob = (1.0 - profile_prob - same_prob - uniform_prob)
    assert mask_prob >= 0.0

    categorical_probs = torch.nn.functional.pad(categorical_probs, pad_shapes, value=mask_prob)

    sh = protein["msa"].shape
    mask_position = torch.rand(sh) < replace_fraction

    bert_msa = shaped_categorical(categorical_probs)
    bert_msa = torch.where(mask_position, bert_msa, protein["msa"])

    # Mix real and masked MSA
    protein["bert_mask"] = mask_position.to(torch.float32)
    protein["true_msa"] = protein["msa"]
    protein["msa"] = bert_msa
    return protein

def nearest_neighbor_clusters(protein, gap_agreement_weight=0.0):
    weights = torch.cat(
        [
            torch.ones(21, device=protein["msa"].device), 
            gap_agreement_weight * torch.ones(1, device=protein["msa"].device),
            torch.zeros(1, device=protein["msa"].device)
        ],
        0,
    )

    # Make agreement score as weighted Hamming distance
    msa_one_hot = make_one_hot(protein["msa"], 23)
    sample_one_hot = protein["msa_mask"][:, :, None] * msa_one_hot
    extra_msa_one_hot = make_one_hot(protein["extra_msa"], 23)
    extra_one_hot = protein["extra_msa_mask"][:, :, None] * extra_msa_one_hot

    num_seq, num_res, _ = sample_one_hot.shape
    extra_num_seq, _, _ = extra_one_hot.shape

    # print(torch.reshape(extra_one_hot, [extra_num_seq, num_res * 23]).shape,
    #     torch.reshape(sample_one_hot * weights, [num_seq, num_res * 23]).transpose(0, 1).shape)
    # Compute tf.einsum('mrc,nrc,c->mn', sample_one_hot, extra_one_hot, weights)
    # in an optimized fashion to avoid possible memory or computation blowup.
    agreement = torch.matmul(
        torch.reshape(extra_one_hot, [extra_num_seq, num_res * 23]),
        torch.reshape(sample_one_hot * weights, [num_seq, num_res * 23]).transpose(0, 1),
    )
    # Assign each sequence in the extra sequences to the closest MSA sample
    protein["extra_cluster_assignment"] = torch.argmax(agreement, dim=1).to(
        torch.int64
    )
    return protein

def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    Computes the sum along segments of a tensor. Similar to 
    tf.unsorted_segment_sum, but only supports 1-D indices.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The 1-D segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """
    assert (
        len(segment_ids.shape) == 1 and
        segment_ids.shape[0] == data.shape[0]
    )
    segment_ids = segment_ids.view(segment_ids.shape[0], *((1,) * len(data.shape[1:])))
    segment_ids = segment_ids.expand(data.shape)
    shape = [num_segments] + list(data.shape[1:])
    tensor = (torch.zeros(*shape, device=segment_ids.device).scatter_add_(0, segment_ids, data.float()))
    tensor = tensor.type(data.dtype)
    return tensor

def summarize_clusters(protein):
    """Produce profile and deletion_matrix_mean within each cluster."""
    num_seq = protein["msa"].shape[0]

    def csum(x):
        return unsorted_segment_sum(x, protein["extra_cluster_assignment"], num_seq)

    mask = protein["extra_msa_mask"]
    mask_counts = 1e-6 + protein["msa_mask"] + csum(mask)  # Include center

    msa_sum = csum(mask[:, :, None] * make_one_hot(protein["extra_msa"], 23))
    msa_sum += make_one_hot(protein["msa"], 23)  # Original sequence
    protein["cluster_profile"] = msa_sum / mask_counts[:, :, None]
    del msa_sum

    del_sum = csum(mask * protein["extra_deletion_matrix"])
    del_sum += protein["deletion_matrix"]  # Original sequence
    protein["cluster_deletion_mean"] = del_sum / mask_counts
    del del_sum
    
    return protein

def make_msa_feat(protein):
    """Create and concatenate MSA features."""
    # Whether there is a domain break. Always zero for chains, but keeping for
    # compatibility with domain datasets.

    protein["between_segment_residues"] = torch.tensor(np.zeros((protein["msa"].shape[1],), dtype=np.int32))

    has_break = torch.clip(protein["between_segment_residues"].to(torch.float32), 0, 1)
    aatype_1hot = make_one_hot(protein["true_msa"][0], 21)

    target_feat = [torch.unsqueeze(has_break, dim=-1),aatype_1hot]

    msa_1hot = make_one_hot(protein["msa"], 23)
    has_deletion = torch.clip(protein["deletion_matrix"], 0.0, 1.0)
    deletion_value = torch.atan(protein["deletion_matrix"] / 3.0) * (2.0 / np.pi)

    msa_feat = [msa_1hot,torch.unsqueeze(has_deletion, dim=-1),torch.unsqueeze(deletion_value, dim=-1),]

    if "cluster_profile" in protein:
        deletion_mean_value = torch.atan(protein["cluster_deletion_mean"] / 3.0) * (2.0 / np.pi)
        msa_feat.extend([protein["cluster_profile"],torch.unsqueeze(deletion_mean_value, dim=-1),])

    if "extra_deletion_matrix" in protein:
        protein["extra_has_deletion"] = torch.clip(protein["extra_deletion_matrix"], 0.0, 1.0)
        protein["extra_deletion_value"] = torch.atan(protein["extra_deletion_matrix"] / 3.0) * (2.0 / np.pi)

    protein["msa_feat"] = torch.cat(msa_feat, dim=-1)
    protein["target_feat"] = torch.cat(target_feat, dim=-1)
    return protein

