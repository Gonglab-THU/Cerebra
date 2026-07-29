import torch
import torch.nn.functional as F


class LieOps:

    
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

        q_norm = torch.norm(q, dim=-1, keepdim=True)
        safe_q_norm = torch.where(q_norm < eps, torch.ones_like(q_norm), q_norm)
        q = q / safe_q_norm

        qw = q[..., 0].clamp(-1.0 + eps, 1.0 - eps)
        xyz = q[..., 1:]
        
        theta = 2.0 * torch.acos(qw)
        sin_half_theta = torch.sqrt(torch.clamp(1.0 - qw**2, min=eps))
  

        safe_sin_half_theta = torch.where(
            sin_half_theta < eps, 
            torch.ones_like(sin_half_theta), 
            sin_half_theta
        )
        
        scale_normal = theta / safe_sin_half_theta
        scale_taylor = 2.0 + (1.0 - qw**2) / 3.0
        
        scale = torch.where(sin_half_theta < eps, scale_taylor, scale_normal)
        return scale.unsqueeze(-1) * xyz

    @staticmethod
    def exp_map(v, eps=1e-6):
        theta = torch.norm(v, dim=-1, keepdim=True)
        
        # stop torch.where  0*NaN gradient propagation
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