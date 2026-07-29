import torch
import torch.nn.functional as F

# class LieOps:
#     """包含了用于 SO(3) 注意力插值的核心数学运算"""
    
#     @staticmethod
#     def quat_multiply(q, r):
#         """四元数乘法 (Batch, ..., 4) * (Batch, ..., 4)"""
#         # q: [w, x, y, z], r: [w, x, y, z]
#         qw, qx, qy, qz = q.unbind(-1)
#         rw, rx, ry, rz = r.unbind(-1)
#         # 汉密尔顿积
#         return torch.stack([
#             qw*rw - qx*rx - qy*ry - qz*rz,
#             qw*rx + qx*rw + qy*rz - qz*ry,
#             qw*ry - qx*rz + qy*rw + qz*rx,
#             qw*rz + qx*ry - qy*rx + qz*rw
#         ], dim=-1)

#     @staticmethod
#     def quat_inv(q):
#         """求逆（共轭），假设已归一化"""
#         return q * torch.tensor([1, -1, -1, -1], device=q.device, dtype=q.dtype)

#     @staticmethod
#     def log_map(q, eps=1e-6):
#         """从四元数映射到李代数 (旋转向量)"""
#         # 强制归一化以防万一
#         q = F.normalize(q, dim=-1)
#         qw = q[..., 0].clamp(-1.0 + eps, 1.0 - eps)
#         xyz = q[..., 1:]
        
#         theta = 2.0 * torch.acos(qw)
#         sin_half_theta = torch.sqrt(1.0 - qw**2)
        
#         # 泰勒展开处理小角度 (sin(x)/x 在 x->0 时不稳定)
#         scale = torch.where(
#             sin_half_theta < eps,
#             2.0 + (1.0 - qw**2) / 3.0, # 泰勒近似
#             theta / sin_half_theta
#         )
#         return scale.unsqueeze(-1) * xyz

#     @staticmethod
#     def exp_map(v, eps=1e-6):
#         """从李代数映射回四元数"""
#         theta = torch.norm(v, dim=-1, keepdim=True)
        
#         # 泰勒展开处理小角度
#         scale = torch.where(
#             theta < eps,
#             0.5 - (theta**2) / 48.0,
#             torch.sin(theta / 2) / theta
#         )
#         qw = torch.cos(theta / 2)
#         qxyz = scale * v
#         return torch.cat([qw, qxyz], dim=-1)

# import torch
# import torch.nn.functional as F

class LieOps:
    """包含了用于 SO(3) 注意力插值的核心数学运算 (防弹级防 NaN 版)"""
    
    @staticmethod
    def quat_multiply(q, r):
        qw, qx, qy, qz = q.unbind(-1)
        rw, rx, ry, rz = r.unbind(-1)
        return torch.stack([
            qw*rw - qx*rx - qy*ry - qz*rz,
            qw*rx + qx*rw + qy*rz - qz*ry,
            qw*ry - qx*rz + qy*rw + qz*rx,
            qw*rz + qx*ry - qy*rx + qz*rw
        ], dim=-1)

    @staticmethod
    def quat_inv(q):
        return q * torch.tensor([1, -1, -1, -1], device=q.device, dtype=q.dtype)

    @staticmethod
    def log_map(q, eps=1e-6):
        # 1. 强力归一化防 NaN
        # 如果输入全0，F.normalize 默认返回全0，我们需要避免这种情况
        q_norm = torch.norm(q, dim=-1, keepdim=True)
        safe_q_norm = torch.where(q_norm < eps, torch.ones_like(q_norm), q_norm)
        q = q / safe_q_norm
        
        # 2. 截断防 acos 导数爆炸
        qw = q[..., 0].clamp(-1.0 + eps, 1.0 - eps)
        xyz = q[..., 1:]
        
        theta = 2.0 * torch.acos(qw)
        sin_half_theta = torch.sqrt(torch.clamp(1.0 - qw**2, min=eps)) # clamp 防止负数
        
        # 3. 阻断 torch.where 的 0*NaN 梯度传播陷阱
        # 我们创建一个 "安全的除数"，即使在极小角度下，除数也不会是 0
        safe_sin_half_theta = torch.where(
            sin_half_theta < eps, 
            torch.ones_like(sin_half_theta), # 如果极小，用 1 代替以安全完成除法
            sin_half_theta
        )
        
        scale_normal = theta / safe_sin_half_theta
        scale_taylor = 2.0 + (1.0 - qw**2) / 3.0
        
        scale = torch.where(sin_half_theta < eps, scale_taylor, scale_normal)
        return scale.unsqueeze(-1) * xyz

    @staticmethod
    def exp_map(v, eps=1e-6):
        theta = torch.norm(v, dim=-1, keepdim=True)
        
        # 阻断 torch.where 的 0*NaN 梯度传播陷阱
        safe_theta = torch.where(
            theta < eps, 
            torch.ones_like(theta), 
            theta
        )
        
        scale_normal = torch.sin(theta / 2) / safe_theta
        scale_taylor = 0.5 - (theta**2) / 48.0
        
        scale = torch.where(theta < eps, scale_taylor, scale_normal)
        
        qw = torch.cos(theta / 2)
        qxyz = scale * v
        return torch.cat([qw, qxyz], dim=-1)