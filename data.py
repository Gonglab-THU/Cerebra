import numpy as np
import torch
import re
from pathlib import Path
from typing import Dict, List, Mapping, Tuple
from functools import reduce
from operator import add
import h5py
from cerebra_model.np import residue_constants
eps = 1e-7

HHBLITS_AA_TO_ID = {
    "A": 0,
    "B": 2,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "J": 20,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "O": 20,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "U": 1,
    "V": 17,
    "W": 18,
    "X": 20,
    "Y": 19,
    "Z": 3,
    "-": 21,
}

ESM_AA_NORMALIZATION = {
    "B": "D",
    "J": "X",
    "O": "X",
    "U": "C",
    "Z": "E",
}
ESM_ALLOWED_AAS = set("ACDEFGHIKLMNPQRSTVWYX")

def NormVec(V):
    axis_x = V[:, 2] - V[:, 1]
    axis_x /= (torch.norm(axis_x, dim=-1).unsqueeze(1) + eps)
    axis_y = V[:, 0] - V[:, 1]
    axis_z = torch.cross(axis_x, axis_y, dim=1)
    axis_z /= (torch.norm(axis_z, dim=-1).unsqueeze(1) + eps)
    axis_y = torch.cross(axis_z, axis_x, dim=1)
    axis_y /= (torch.norm(axis_y, dim=-1).unsqueeze(1) + eps)
    Vec = torch.stack([axis_x, axis_y, axis_z], dim=1)
    return Vec

def QuaternionMM(q1, q2):
    a = q1[..., 0] * q2[..., 0] - (q1[..., 1:] * q2[..., 1:]).sum(-1)
    bcd = torch.cross(q2[..., 1:], q1[..., 1:], dim=-1) + q1[..., 0].unsqueeze(-1) * q2[..., 1:] + q2[..., 0].unsqueeze(-1) * q1[..., 1:]
    q = torch.cat([a.unsqueeze(-1), bcd], dim=-1)
    return q

def NormQuaternionMM(q1, q2):
    q = QuaternionMM(q1, q2)
    return NormQuaternion(q)
    
def TranslationRotation(q, p):
    p4 = torch.cat([torch.zeros_like(p[..., 0]).unsqueeze(-1), p], dim=-1)
    q_1 = torch.cat([q[..., 0].unsqueeze(-1), -q[..., 1:]], dim=-1)
    return QuaternionMM(QuaternionMM(q, p4), q_1)[..., 1:]

def Rotation2Quaternion(r):
    a = torch.sqrt(r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2] + 1)/2.
    b = (r[..., 2, 1] - r[..., 1, 2])/(4 * a)
    c = (r[..., 0, 2] - r[..., 2, 0])/(4 * a)
    d = (r[..., 1, 0] - r[..., 0, 1])/(4 * a)
    q = torch.stack([a, b, c, d], dim=-1)
    q = NormQuaternion(q)
    return q

def NormQuaternion(q):
    q = q/torch.sqrt((q * q).sum(-1, keepdim=True))
    q = torch.sign(torch.sign(q[..., 0]) + 0.5).unsqueeze(-1) * q
    return q

def PsiPhi(atoms):
    def psi(CA, C, N):
        a = N[1:] - C[:-1]
        b = C - CA
        c = N - CA
        ab = torch.cross(a, b[:-1])
        bc = torch.cross(b[:-1], c[:-1])
        ca = torch.cross(c[:-1], a)
        
        cos_ca_b = torch.sum(ca * b[:-1], dim=-1) / (torch.linalg.norm(ca, dim=-1) * torch.linalg.norm(b[:-1], dim=-1) + eps)
        cospsi = torch.sum(ab * bc, dim=-1)/(torch.linalg.norm(ab, dim=-1) * torch.linalg.norm(bc, dim=-1) + eps)
        cospsi = np.pi - torch.arccos(torch.clamp(cospsi, max=1, min=-1))
        return (cos_ca_b / abs(cos_ca_b)) * cospsi

    def phi(CA, C, N):
        b = C - CA
        c = N - CA
        d = C[:-1] - N[1:]
        bc = torch.cross(b[1:], c[1:])
        cd = torch.cross(c[1:], d)
        bd = torch.cross(b[1:], d)
        cos_bd_c = torch.sum(bd * c[1:], dim=-1) / (torch.linalg.norm(bd, dim=-1) * torch.linalg.norm(c[1:], dim=-1) + eps)
        cosphi = torch.sum(bc * cd, dim=-1) / (torch.linalg.norm(bc, dim=-1) * torch.linalg.norm(cd, dim=-1) + eps)
        cosphi = np.pi - torch.arccos(torch.clamp(cosphi, max=1, min=-1))
        return (cos_bd_c / abs(cos_bd_c)) * cosphi
    N, CA, C = atoms[:, 2], atoms[:, 0], atoms[:, 1]
    return torch.stack([psi(CA, C, N), phi(CA, C, N)], dim=1)

def extract_ca_c_n_cb(all_atom_positions, aatype):
    """
    从 all_atom_positions [L, 37, 3] 中提取 CA, C, N, CB 坐标。
    排序要求: CA, C, N, CB -> 最终形状 [L, 4, 3]
    特殊处理: 若残基为甘氨酸 (GLY)，则用 CA 的坐标替代 CB。
    """

    idx_N  = residue_constants.atom_order['N']
    idx_CA = residue_constants.atom_order['CA']
    idx_C  = residue_constants.atom_order['C']
    idx_CB = residue_constants.atom_order['CB']

    pos_N  = all_atom_positions[:, idx_N,  :]
    pos_CA = all_atom_positions[:, idx_CA, :]
    pos_C  = all_atom_positions[:, idx_C,  :]
    pos_CB = all_atom_positions[:, idx_CB, :]

    gly_idx = residue_constants.restype_order['G']
    
    is_gly = (aatype == gly_idx)

    is_gly_expanded = is_gly.unsqueeze(-1)

    pos_CB_modified = torch.where(is_gly_expanded, pos_CA, pos_CB)
    selected_positions = torch.stack([pos_CA, pos_C, pos_N, pos_CB_modified], dim=1)

    return selected_positions

def comp_label(atoms):
    nres = atoms.shape[0]
    N_CA_C = atoms[:, [2, 0, 1], :].reshape(-1, 3, 3)
    rotation = NormVec(N_CA_C)
    U, _, V = torch.svd(torch.eye(3).unsqueeze(0).permute(0, 2, 1) @ rotation)
    d = torch.sign(torch.det(U @ V.permute(0, 2, 1)))
    Id = torch.eye(3).repeat(nres, 1, 1)
    Id[:, 2, 2] = d
    r = V @ (Id @ U.permute(0, 2, 1))
    q = Rotation2Quaternion(r)
    q_1 = torch.cat([q[..., 0].unsqueeze(-1), -q[..., 1:]], dim=-1)
    QAll = NormQuaternionMM(q.unsqueeze(1).repeat(1, nres, 1), q_1.unsqueeze(0).repeat(nres, 1, 1))
    
    QAll[..., 0][torch.isnan(QAll[..., 0])] = 1.
    QAll[torch.isnan(QAll)] = 0.
    QAll = NormQuaternion(QAll)
    
    xyz_CA = torch.einsum('a b i, a i j -> a b j', atoms[:, 0].unsqueeze(0) - atoms[:, 0].unsqueeze(1), r)
    xyz_C  = torch.einsum('a b i, a i j -> a b j', atoms[:, 1].unsqueeze(0) - atoms[:, 0].unsqueeze(1), r)
    xyz_N  = torch.einsum('a b i, a i j -> a b j', atoms[:, 2].unsqueeze(0) - atoms[:, 0].unsqueeze(1), r)
    xyz_CB = torch.einsum('a b i, a i j -> a b j', atoms[:, 3].unsqueeze(0) - atoms[:, 0].unsqueeze(1), r)
    
    CA_C_N_CB = torch.stack([xyz_CA, xyz_C, xyz_N, xyz_CB], dim=-2)
    
    CA = atoms[:, 0].unsqueeze(0) - atoms[:, 0].unsqueeze(1)
    r_CA = torch.sqrt((CA * CA).sum(-1) + eps)
    CB_dist = atoms[:, 3, :].unsqueeze(0) - atoms[:, 3, :].unsqueeze(1)
    CB_dist = torch.sqrt((CB_dist * CB_dist).sum(-1))
    CB_dist = (CB_dist*2 - 7).long()
    CB_dist = CB_dist.clamp(min=0, max=35)
    psi_phi = PsiPhi(atoms)

    return CA_C_N_CB, CB_dist, r_CA, QAll, psi_phi

FeatureDict = Mapping[str, np.ndarray]

def make_msa_features(msa) -> FeatureDict:
    unique_msa, inverse_indices = torch.unique(msa, sorted=False, return_inverse=True, dim=0)
    t1 = unique_msa[0].clone()
    unique_msa[0] = msa[0]
    unique_msa[inverse_indices[0]] = t1

    features = {}
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

def sample_msa(protein, max_seq, keep_extra=True, seed=42):
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


def sample_train_max_seq(max_seq_cap=128):
    # Biased sampling: keep full-depth frequently, but expose low-depth cases.
    candidates = np.array([1,8, 16, 32, 64, 128], dtype=np.int64)
    probs = np.array([0.01, 0.05, 0.09, 0.2, 0.25, 0.4], dtype=np.float64)
    sampled = int(np.random.choice(candidates, p=probs))
    return min(sampled, max_seq_cap)


def repeat_to_rows(x, target_rows):
    cur_rows = x.shape[0]
    if cur_rows >= target_rows:
        return x[:target_rows]
    repeat_num = int(target_rows / cur_rows) + 1
    reps = [repeat_num] + [1] * (x.dim() - 1)
    return x.repeat(*reps)[:target_rows]

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


def sanitize_name(name: str) -> str:
    name = name.strip().split()[0] if name.strip() else "query"
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("._") or "query"


def normalize_target_sequence(sequence: str) -> str:
    clean = []
    for char in sequence:
        if char.isspace() or char == "-":
            continue
        aa = ESM_AA_NORMALIZATION.get(char.upper(), char.upper())
        clean.append(aa if aa in ESM_ALLOWED_AAS else "X")
    if not clean:
        raise ValueError("Input sequence is empty after cleaning.")
    return "".join(clean)


def read_fasta_records(path) -> List[Tuple[str, str]]:
    path = Path(path)
    records: List[Tuple[str, str]] = []
    name = None
    seq_lines: List[str] = []

    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(seq_lines)))
                name = line[1:].strip() or path.stem
                seq_lines = []
            else:
                if name is None:
                    name = path.stem
                seq_lines.append(line)

    if name is not None:
        records.append((name, "".join(seq_lines)))
    return records


def msa_row_to_ids(row: str) -> List[int]:
    return [HHBLITS_AA_TO_ID.get(char.upper(), 20) for char in row]


def read_fasta_input(path) -> Tuple[str, str, torch.Tensor]:
    records = read_fasta_records(path)
    if len(records) != 1:
        raise ValueError(
            f"{path} contains {len(records)} FASTA records; this script expects one target per run."
        )

    name, sequence = records[0]
    target_sequence = normalize_target_sequence(sequence)
    msa = torch.LongTensor([msa_row_to_ids(target_sequence)])
    return sanitize_name(name), target_sequence, msa


def remove_a3m_insertions(sequence: str) -> str:
    return "".join(char for char in sequence.strip() if not char.islower() and char != ".")


def read_a3m_input(path) -> Tuple[str, str, torch.Tensor]:
    path = Path(path)
    records = read_fasta_records(path)
    if not records:
        raise ValueError(f"No aligned sequences found in {path}.")

    name = sanitize_name(records[0][0])
    aligned_rows = [remove_a3m_insertions(seq) for _, seq in records]
    lengths = {len(row) for row in aligned_rows}
    if len(lengths) != 1:
        raise ValueError(
            f"A3M rows have inconsistent aligned lengths after removing insertions: {sorted(lengths)}"
        )

    target_columns = [i for i, aa in enumerate(aligned_rows[0]) if aa != "-"]
    if not target_columns:
        raise ValueError("The first A3M sequence has no non-gap target columns.")

    aligned_rows = ["".join(row[i] for i in target_columns) for row in aligned_rows]
    target_sequence = normalize_target_sequence(aligned_rows[0])
    if len(target_sequence) != len(aligned_rows[0]):
        raise ValueError("A3M target sequence length changed during cleaning.")

    msa = torch.LongTensor([msa_row_to_ids(row) for row in aligned_rows])
    return name, target_sequence, msa


def filter_msa_rows(msa: torch.Tensor, max_msa_sequences: int) -> torch.Tensor:
    if max_msa_sequences <= 0 or msa.shape[0] <= max_msa_sequences:
        return msa

    non_gap_ratio = (msa < 21).float().sum(dim=1) / msa.shape[1]
    keep = [0]
    informative = [
        idx
        for idx in torch.argsort(non_gap_ratio, descending=True).tolist()
        if idx != 0 and non_gap_ratio[idx] > 0.25
    ]
    keep.extend(informative)
    if len(keep) < max_msa_sequences:
        keep.extend(idx for idx in range(1, msa.shape[0]) if idx not in keep)
    keep = keep[:max_msa_sequences]
    keep_tensor = torch.as_tensor(keep, dtype=torch.long, device=msa.device)
    return msa[keep_tensor]


def build_model_features(
    msa: torch.Tensor,
    target_sequence: str,
    esm2_embedding: torch.Tensor,
    cycles: int,
    seed: int,
    max_msa_sequences: int,
    sample_msa_sequences: int,
    min_msa_rows: int,
) -> Dict[str, torch.Tensor]:
    if cycles <= 0:
        raise ValueError("cycles must be positive.")
    if sample_msa_sequences <= 0:
        raise ValueError("sample_msa_sequences must be positive.")
    if min_msa_rows <= 0:
        raise ValueError("min_msa_rows must be positive.")
    if msa.shape[1] != len(target_sequence):
        raise ValueError(
            f"MSA length ({msa.shape[1]}) does not match target length ({len(target_sequence)})."
        )
    if esm2_embedding.shape[0] != len(target_sequence):
        raise ValueError(
            f"ESM2 length ({esm2_embedding.shape[0]}) does not match target length ({len(target_sequence)})."
        )

    msa = filter_msa_rows(msa, max_msa_sequences)
    nonensemble_feat = make_msa_features(msa)
    nonensemble_feat = make_hhblits_profile(nonensemble_feat)
    nonensemble_feat = make_msa_mask(nonensemble_feat)

    msa_feat = []
    target_feat = []
    msa_depth = []
    for cycle in range(cycles):
        msa_features = sample_msa(
            nonensemble_feat.copy(),
            sample_msa_sequences,
            seed=seed + cycle,
        )
        msa_depth.append(min(msa_features["msa"].shape[0], min_msa_rows) / float(min_msa_rows))
        msa_features = make_masked_msa(msa_features, "eval")
        msa_features = nearest_neighbor_clusters(msa_features)
        msa_features = summarize_clusters(msa_features)
        msa_features = make_msa_feat(msa_features)
        msa_feat.append(msa_features["msa_feat"])
        target_feat.append(msa_features["target_feat"])

    msa_feat_tensor = torch.stack(msa_feat)
    true_msa = msa_features["true_msa"]
    bert_mask = msa_features["bert_mask"]

    msa_num, seq_length = true_msa.shape
    if msa_num < min_msa_rows:
        repeat_num = int(min_msa_rows / msa_num) + 1
        feature_dim = msa_feat_tensor.shape[-1]
        msa_feat_tensor = (
            msa_feat_tensor[:, None, :, :, :]
            .repeat(1, repeat_num, 1, 1, 1)
            .reshape(cycles, -1, seq_length, feature_dim)[:, :min_msa_rows]
        )
        true_msa = true_msa[None].repeat(repeat_num, 1, 1).reshape(-1, seq_length)[:min_msa_rows]
        bert_mask = bert_mask[None].repeat(repeat_num, 1, 1).reshape(-1, seq_length)[:min_msa_rows]

    return {
        "msa_feat": msa_feat_tensor,
        "true_msa": true_msa,
        "bert_mask": bert_mask,
        "esm2": esm2_embedding.to(dtype=torch.float32),
        "msa_depth": torch.tensor(msa_depth, dtype=torch.float32),
        "target_feat": torch.stack(target_feat),
        "residue_index": torch.arange(seq_length)[None].repeat(cycles, 1),
        "mask": torch.ones(seq_length, dtype=torch.float32),
        "seq": target_sequence,
    }

def get_feature(msa_file, esm2_file, cycle_num, max_crop, mode):
    with h5py.File(esm2_file, 'r') as fin:
        esm2 = torch.tensor(fin['X1D'][:])


    fin = h5py.File(msa_file, 'r')
    msa = torch.tensor(fin['toks'][:])
    # if mode == 'eval':
    #     msa_tmp = []
    #     if msa.shape[0] > 5120:
    #         # gap_ratio = (msa == 21).long().sum(1)
    #         msa = msa[:5120]


    atoms = torch.tensor(fin['xyz'][:])
    mask = torch.tensor(fin['mask'][:])
    seq  = fin['seq'][0].decode('utf-8')

    fin.close()

    original_length = mask.shape[0]
    valid_mask = mask > 0
    residue_index = torch.arange(original_length, dtype=torch.int64)

    if mode == 'train':
        esm2 = esm2[valid_mask]
    msa = msa.transpose(1, 0)[valid_mask].transpose(1, 0)
    atoms = atoms[valid_mask]
    if mode == 'eval':
        esm2 = esm2[valid_mask]
    mask = mask[valid_mask]
    residue_index = residue_index[valid_mask]
    
    seq_mask = []
    for idx, m in enumerate(mask.tolist()):
        if m > 0:
            seq_mask.append(seq[idx])
    seq = ''.join(seq_mask)

    pdb_length = msa.shape[1]
    c1, c2 = 0, pdb_length
    if mode == 'train':
        if pdb_length - max_crop == 0:
            c1 = 0
        else:
            c1 = np.random.randint(pdb_length - max_crop)
    c2 = c1 + max_crop
    msa = msa[:, c1:c2]
    atoms = atoms[c1:c2]
    mask = mask[c1:c2]
    seq = seq[c1:c2]
    esm2 = esm2[c1:c2]
    residue_index = residue_index[c1:c2]

    nonensemble_feat = make_msa_features(msa)
    nonensemble_feat = make_hhblits_profile(nonensemble_feat)
    nonensemble_feat = make_msa_mask(nonensemble_feat)
    

    msa_feat, target_feat = [], []
    msa_depth = []
    for cycle in range(cycle_num):
        if mode == 'train':
            max_seq = sample_train_max_seq(128)
        else:
            max_seq = 512
        msa_features = sample_msa(nonensemble_feat.copy(), max_seq)
        msa_depth.append(min(msa_features["msa"].shape[0], 128) / 128.0)
        msa_features = make_masked_msa(msa_features, mode)
        msa_features = nearest_neighbor_clusters(msa_features)
        msa_features = summarize_clusters(msa_features)
        msa_features = make_msa_feat(msa_features)
        cycle_msa_feat = msa_features['msa_feat']
        if mode == 'train' and cycle_msa_feat.shape[0] < 128:
            cycle_msa_feat = repeat_to_rows(cycle_msa_feat, 128)
        msa_feat.append(cycle_msa_feat)
        target_feat.append(msa_features['target_feat'])
    
    
    msa_feat = torch.stack(msa_feat)
    true_msa = msa_features['true_msa']
    bert_mask = msa_features['bert_mask']

    msa_num, seq_length = true_msa.shape 
    if msa_num < 128 and mode == 'train':
        repeat_num = int(128/msa_num) + 1
        msa_feat = msa_feat[:, None, :, :, :].repeat(1, repeat_num, 1, 1, 1).reshape(cycle_num, -1, seq_length, 49)[:, :128]
        true_msa = true_msa[None].repeat(repeat_num, 1, 1).reshape(-1, seq_length)[:128]
        bert_mask = bert_mask[None].repeat(repeat_num, 1, 1).reshape(-1, seq_length)[:128]
    # if msa_num < 128 and mode == 'eval':
    #     repeat_num = int(128/msa_num) + 1
    #     msa_feat = msa_feat[:, None, :, :, :].repeat(1, repeat_num, 1, 1, 1).reshape(cycle_num, -1, seq_length, 49)[:, :128]
    #     true_msa = true_msa[None].repeat(repeat_num, 1, 1).reshape(-1, seq_length)[:128]
    #     bert_mask = bert_mask[None].repeat(repeat_num, 1, 1).reshape(-1, seq_length)[:128]

    ret_feature = {}
    ret_feature['msa_feat'] = msa_feat
    ret_feature['true_msa'] = true_msa
    ret_feature['bert_mask'] = bert_mask
    ret_feature['esm2'] = esm2
    ret_feature['msa_depth'] = torch.tensor(msa_depth, dtype=torch.float32)
    ret_feature['target_feat'] = torch.stack(target_feat)
    ret_feature['residue_index'] = residue_index[None].repeat(cycle_num, 1)
    

    CA_C_N_CB, CB_dist, r_CA, QAll, psi_phi = comp_label(atoms)
    ret_feature['xyz'] = CA_C_N_CB
    ret_feature['CB_dist'] = CB_dist
    ret_feature['quaternion'] = QAll
    ret_feature['psi_phi'] = psi_phi
    ret_feature['mask'] = mask
    ret_feature['atoms'] = atoms
    ret_feature['seq'] = seq
    return ret_feature

def get_feature_extend(msa_file, esm2_file, all_atom_file, cycle_num, max_crop, mode):
    with h5py.File(esm2_file, 'r') as fin:
        esm2 = torch.tensor(fin['X1D'][:])
    if all_atom_file[:7] == '/mydata':
        with h5py.File(all_atom_file, 'r') as fin:
            if 'all_atoms_coord' in fin.keys():
                all_atoms_pos = torch.tensor(fin['all_atoms_coord'][:])
                all_atoms_mask = torch.tensor(fin['all_atoms_mask'][:])
            else:
                all_atoms_pos = torch.tensor(fin['all_atom_positions'][:])
                all_atoms_mask = torch.tensor(fin['all_atom_mask'][:])
                openfold_seq = torch.tensor(fin['aatype'][:],dtype=torch.long)
    else:
        with h5py.File(all_atom_file, 'r') as fin:
            all_atoms_pos = torch.tensor(fin['all_atoms_pos'][:])
            all_atoms_mask = torch.tensor(fin['all_atoms_mask'][:])

    fin = h5py.File(msa_file, 'r')
    msa = torch.tensor(fin['toks'][:])
    seq  = fin['seq'][0].decode('utf-8')
    # if mode == 'eval':
    #     msa_tmp = []
    #     if msa.shape[0] > 5120:
    #         # gap_ratio = (msa == 21).long().sum(1)
    #         msa = msa[:5120]

    if 'xyz' in fin.keys():
        atoms = torch.tensor(fin['xyz'][:])
        mask = torch.tensor(fin['mask'][:])
    else:
        atoms = extract_ca_c_n_cb(all_atoms_pos, openfold_seq)
        mask = all_atoms_mask[:,0]*all_atoms_mask[:,1]*all_atoms_mask[:,2]

    fin.close()
    
    original_length = mask.shape[0]
    valid_mask = mask > 0
    residue_index = torch.arange(original_length, dtype=torch.int64)

    if mode == 'train':
        esm2 = esm2[valid_mask]
    msa = msa.transpose(1, 0)[valid_mask].transpose(1, 0)
    atoms = atoms[valid_mask]
    all_atoms_pos = all_atoms_pos[valid_mask]
    all_atoms_mask = all_atoms_mask[valid_mask]
    if mode == 'eval':
        esm2 = esm2[valid_mask]
    mask = mask[valid_mask]
    residue_index = residue_index[valid_mask]
    
    seq_mask = []
    for idx, m in enumerate(mask.tolist()):
        if m > 0:
            seq_mask.append(seq[idx])
    seq = ''.join(seq_mask)

    pdb_length = msa.shape[1]
    c1, c2 = 0, pdb_length
    if mode == 'train':
        if pdb_length - max_crop == 0:
            c1 = 0
        else:
            c1 = np.random.randint(pdb_length - max_crop)
    c2 = c1 + max_crop
    msa = msa[:, c1:c2]
    atoms = atoms[c1:c2]
    mask = mask[c1:c2]
    seq = seq[c1:c2]
    esm2 = esm2[c1:c2]
    all_atoms_pos =all_atoms_pos[c1:c2]
    all_atoms_mask = all_atoms_mask[c1:c2]
    residue_index = residue_index[c1:c2]
    nonensemble_feat = make_msa_features(msa)
    nonensemble_feat = make_hhblits_profile(nonensemble_feat)
    nonensemble_feat = make_msa_mask(nonensemble_feat)
    

    msa_feat, target_feat = [], []
    msa_depth = []
    for cycle in range(cycle_num):
        if mode == 'train':
            max_seq = sample_train_max_seq(128)
        else:
            max_seq = 512
        msa_features = sample_msa(nonensemble_feat.copy(), max_seq)
        msa_depth.append(min(msa_features["msa"].shape[0], 128) / 128.0)
        msa_features = make_masked_msa(msa_features, mode)
        msa_features = nearest_neighbor_clusters(msa_features)
        msa_features = summarize_clusters(msa_features)
        msa_features = make_msa_feat(msa_features)
        cycle_msa_feat = msa_features['msa_feat']
        if mode == 'train' and cycle_msa_feat.shape[0] < 128:
            cycle_msa_feat = repeat_to_rows(cycle_msa_feat, 128)
        msa_feat.append(cycle_msa_feat)
        target_feat.append(msa_features['target_feat'])
    
    
    msa_feat = torch.stack(msa_feat)
    true_msa = msa_features['true_msa']
    bert_mask = msa_features['bert_mask']

    msa_num, seq_length = true_msa.shape 
    if msa_num < 128 and mode == 'train':
        repeat_num = int(128/msa_num) + 1
        msa_feat = msa_feat[:, None, :, :, :].repeat(1, repeat_num, 1, 1, 1).reshape(cycle_num, -1, seq_length, 49)[:, :128]
        true_msa = true_msa[None].repeat(repeat_num, 1, 1).reshape(-1, seq_length)[:128]
        bert_mask = bert_mask[None].repeat(repeat_num, 1, 1).reshape(-1, seq_length)[:128]
    # if msa_num < 128 and mode == 'eval':
    #     repeat_num = int(128/msa_num) + 1
    #     msa_feat = msa_feat[:, None, :, :, :].repeat(1, repeat_num, 1, 1, 1).reshape(cycle_num, -1, seq_length, 49)[:, :128]
    #     true_msa = true_msa[None].repeat(repeat_num, 1, 1).reshape(-1, seq_length)[:128]
    #     bert_mask = bert_mask[None].repeat(repeat_num, 1, 1).reshape(-1, seq_length)[:128]

    ret_feature = {}
    ret_feature['msa_feat'] = msa_feat
    ret_feature['true_msa'] = true_msa
    ret_feature['bert_mask'] = bert_mask
    ret_feature['esm2'] = esm2
    ret_feature['msa_depth'] = torch.tensor(msa_depth, dtype=torch.float32)
    ret_feature['target_feat'] = torch.stack(target_feat)
    ret_feature['residue_index'] = residue_index[None].repeat(cycle_num, 1)
    

    CA_C_N_CB, CB_dist, r_CA, QAll, psi_phi = comp_label(atoms)
    ret_feature['xyz'] = CA_C_N_CB
    ret_feature['CB_dist'] = CB_dist
    ret_feature['quaternion'] = QAll
    ret_feature['psi_phi'] = psi_phi
    ret_feature['mask'] = mask
    ret_feature['atoms'] = atoms
    ret_feature['seq'] = seq
    ret_feature['all_atoms_mask'] = all_atoms_mask
    ret_feature['all_atoms_pos'] = all_atoms_pos
    return ret_feature

def _read_all_atom_feature_file(all_atom_file):
    with h5py.File(all_atom_file, 'r') as fin:
        if 'all_atoms_coord' in fin.keys():
            all_atoms_pos = torch.tensor(fin['all_atoms_coord'][:])
        elif 'all_atom_positions' in fin.keys():
            all_atoms_pos = torch.tensor(fin['all_atom_positions'][:])
        elif 'all_atoms_pos' in fin.keys():
            all_atoms_pos = torch.tensor(fin['all_atoms_pos'][:])
        elif 'all_atom_pos' in fin.keys():
            all_atoms_pos = torch.tensor(fin['all_atom_pos'][:])
        else:
            raise KeyError(f'Cannot find all atom positions in {all_atom_file}')

        if 'all_atoms_mask' in fin.keys():
            all_atoms_mask = torch.tensor(fin['all_atoms_mask'][:])
        elif 'all_atom_mask' in fin.keys():
            all_atoms_mask = torch.tensor(fin['all_atom_mask'][:])
        else:
            all_atoms_mask = torch.ones(all_atoms_pos.shape[:-1], dtype=all_atoms_pos.dtype)

        openfold_seq = None
        if 'aatype' in fin.keys():
            openfold_seq = torch.tensor(fin['aatype'][:], dtype=torch.long)

    return all_atoms_pos, all_atoms_mask, openfold_seq


def _extract_backbone_atoms(all_atoms_pos, all_atoms_mask=None, openfold_seq=None):
    if all_atoms_pos.shape[1] >= 37:
        if openfold_seq is not None:
            return extract_ca_c_n_cb(all_atoms_pos, openfold_seq)
        idx_N = residue_constants.atom_order['N']
        idx_CA = residue_constants.atom_order['CA']
        idx_C = residue_constants.atom_order['C']
        idx_CB = residue_constants.atom_order['CB']
        pos_N = all_atoms_pos[:, idx_N, :]
        pos_CA = all_atoms_pos[:, idx_CA, :]
        pos_C = all_atoms_pos[:, idx_C, :]
        pos_CB = all_atoms_pos[:, idx_CB, :]
        if all_atoms_mask is not None and all_atoms_mask.shape[1] > idx_CB:
            cb_mask = all_atoms_mask[:, idx_CB].to(dtype=torch.bool).unsqueeze(-1)
            pos_CB = torch.where(cb_mask, pos_CB, pos_CA)
        return torch.stack([pos_CA, pos_C, pos_N, pos_CB], dim=1)

    if all_atoms_pos.shape[1] >= 5:
        pos_N = all_atoms_pos[:, 0, :]
        pos_CA = all_atoms_pos[:, 1, :]
        pos_C = all_atoms_pos[:, 2, :]
        pos_CB = all_atoms_pos[:, 4, :]
        if all_atoms_mask is not None and all_atoms_mask.shape[1] > 4:
            cb_mask = all_atoms_mask[:, 4].to(dtype=torch.bool).unsqueeze(-1)
            pos_CB = torch.where(cb_mask, pos_CB, pos_CA)
        elif openfold_seq is not None:
            gly_idx = residue_constants.restype_order['G']
            is_gly = (openfold_seq == gly_idx).unsqueeze(-1)
            pos_CB = torch.where(is_gly, pos_CA, pos_CB)
        return torch.stack([pos_CA, pos_C, pos_N, pos_CB], dim=1)

    return all_atoms_pos[:, :4, :]


def _make_backbone_mask(all_atoms_pos, all_atoms_mask):
    if all_atoms_mask.dim() == 1:
        return all_atoms_mask
    if all_atoms_pos.shape[1] >= 37:
        idx_N = residue_constants.atom_order['N']
        idx_CA = residue_constants.atom_order['CA']
        idx_C = residue_constants.atom_order['C']
        return all_atoms_mask[:, idx_N] * all_atoms_mask[:, idx_CA] * all_atoms_mask[:, idx_C]
    if all_atoms_mask.shape[1] >= 3:
        return all_atoms_mask[:, 0] * all_atoms_mask[:, 1] * all_atoms_mask[:, 2]
    return all_atoms_mask[:, 0]


def get_a3m_feature(msa_file, esm2_file, all_atom_file=None, cycle_num=4, device=None, max_crop=None, mode='eval'):
    if isinstance(all_atom_file, int):
        device = cycle_num
        cycle_num = all_atom_file
        all_atom_file = None

    _, target_seq, msa = read_a3m_input(msa_file)
    with h5py.File(esm2_file,'r') as f:
        if 'X1D' in f.keys():
            esm2 = torch.tensor(f['X1D'][:])
        else:
            esm2 = torch.tensor(f['x1d'][:])

    atoms = None
    mask = torch.ones(msa.shape[1], dtype=torch.float32)
    all_atoms_pos, all_atoms_mask = None, None
    if all_atom_file is not None:
        all_atoms_pos, all_atoms_mask, openfold_seq = _read_all_atom_feature_file(all_atom_file)
        atoms = _extract_backbone_atoms(all_atoms_pos, all_atoms_mask, openfold_seq)
        mask = _make_backbone_mask(all_atoms_pos, all_atoms_mask)

    if msa.shape[0] > 2000:
        msa = filter_msa_rows(msa, 2000)

    if mask.shape[0] != msa.shape[1] or esm2.shape[0] != msa.shape[1]:
        min_length = min(mask.shape[0], msa.shape[1], esm2.shape[0])
        msa = msa[:, :min_length]
        esm2 = esm2[:min_length]
        mask = mask[:min_length]
        target_seq = target_seq[:min_length]
        if atoms is not None:
            atoms = atoms[:min_length]
            all_atoms_pos = all_atoms_pos[:min_length]
            all_atoms_mask = all_atoms_mask[:min_length]

    original_length = mask.shape[0]
    residue_index = torch.arange(original_length, dtype=torch.int64)
    valid_mask = mask > 0
    msa = msa.transpose(1, 0)[valid_mask].transpose(1, 0)
    esm2 = esm2[valid_mask]
    target_seq = ''.join([AA for AA, is_valid in zip(target_seq, valid_mask.tolist()) if is_valid])
    if atoms is not None:
        atoms = atoms[valid_mask]
        all_atoms_pos = all_atoms_pos[valid_mask]
        all_atoms_mask = all_atoms_mask[valid_mask]
    mask = mask[valid_mask]
    residue_index = residue_index[valid_mask]

    pdb_length = msa.shape[1]
    c1, c2 = 0, pdb_length
    if max_crop is None:
        max_crop = pdb_length
    crop_size = min(max_crop, pdb_length)
    if mode == 'train':
        if pdb_length - crop_size == 0:
            c1 = 0
        else:
            c1 = np.random.randint(pdb_length - crop_size)
    c2 = c1 + crop_size
    msa = msa[:, c1:c2]
    mask = mask[c1:c2]
    target_seq = target_seq[c1:c2]
    esm2 = esm2[c1:c2]
    if atoms is not None:
        atoms = atoms[c1:c2]
        all_atoms_pos = all_atoms_pos[c1:c2]
        all_atoms_mask = all_atoms_mask[c1:c2]
    residue_index = residue_index[c1:c2]

    nonensemble_feat = make_msa_features(msa)
    nonensemble_feat = make_hhblits_profile(nonensemble_feat)
    nonensemble_feat = make_msa_mask(nonensemble_feat)

    msa_feat, target_feat = [], []
    msa_depth = []
    for cycle in range(cycle_num):
        if mode == 'train':
            max_seq = sample_train_max_seq(128)
            seed = None
        else:
            max_seq = 512
            seed = 42 + cycle
        msa_features = sample_msa(nonensemble_feat.copy(), max_seq, seed=seed)
        msa_depth.append(min(msa_features["msa"].shape[0], 128) / 128.0)
        msa_features = make_masked_msa(msa_features, mode)
        msa_features = nearest_neighbor_clusters(msa_features)
        msa_features = summarize_clusters(msa_features)
        msa_features = make_msa_feat(msa_features)
        cycle_msa_feat = msa_features['msa_feat']
        if mode == 'train' and cycle_msa_feat.shape[0] < 128:
            cycle_msa_feat = repeat_to_rows(cycle_msa_feat, 128)
        msa_feat.append(cycle_msa_feat)
        target_feat.append(msa_features['target_feat'])
    
    
    msa_feat = torch.stack(msa_feat)
    true_msa = msa_features['true_msa']
    bert_mask = msa_features['bert_mask']

    msa_num, seq_length = true_msa.shape 

    if msa_num < 128 and mode == 'train':
        repeat_num = int(128/msa_num) + 1
        msa_feat = msa_feat[:, None, :, :, :].repeat(1, repeat_num, 1, 1, 1).reshape(cycle_num, -1, seq_length, 49)[:, :128]
        true_msa = true_msa[None].repeat(repeat_num, 1, 1).reshape(-1, seq_length)[:128]
        bert_mask = bert_mask[None].repeat(repeat_num, 1, 1).reshape(-1, seq_length)[:128]

    ret_feature = {}
    ret_feature['msa_feat'] = msa_feat
    ret_feature['true_msa'] = true_msa
    ret_feature['bert_mask'] = bert_mask
    ret_feature['esm2'] = esm2
    ret_feature['msa_depth'] = torch.tensor(msa_depth, dtype=torch.float32)
    ret_feature['target_feat'] = torch.stack(target_feat)
    ret_feature['residue_index'] = residue_index[None].repeat(cycle_num, 1)
    
    if atoms is not None:
        CA_C_N_CB, CB_dist, r_CA, QAll, psi_phi = comp_label(atoms)
        ret_feature['xyz'] = CA_C_N_CB
        ret_feature['CB_dist'] = CB_dist
        ret_feature['quaternion'] = QAll
        ret_feature['psi_phi'] = psi_phi
        ret_feature['atoms'] = atoms
        ret_feature['all_atoms_mask'] = all_atoms_mask
        ret_feature['all_atoms_pos'] = all_atoms_pos
    ret_feature['seq'] = target_seq
    ret_feature['mask'] = mask
    if device is not None:
        ret_feature = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in ret_feature.items()
        }
    return ret_feature
