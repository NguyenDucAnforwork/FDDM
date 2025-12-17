import torch
import torch.nn as nn
import torch.nn.functional as F
from model.bbdm.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel
from model.bbdm.VQGAN.vqgan import VQModel

class LatentFlowMatchingModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        
        # 1. Load VQGAN (Frozen) - Dùng để encode/decode ảnh
        vqgan_config = opt['VQGAN']['params']
        print(f"[INFO] Loading VQGAN from {vqgan_config['ckpt_path']}...")
        self.vqgan = VQModel(**vqgan_config).eval()
        self.vqgan.init_from_ckpt(vqgan_config['ckpt_path'])
        for param in self.vqgan.parameters():
            param.requires_grad = False
            
        # 2. Velocity Estimator (UNet) chạy trên Latent Space
        # Latent của VQGAN-f4 có 3 channels
        # Input = Noisy Latent (3) + Condition Latent (3) = 6 channels
        model_params = opt['flow_matching']['unet_params']
        self.net = UNetModel(**model_params)

    def encode(self, x):
        # Encode ảnh (Batch, 3, H, W) -> Latent (Batch, 3, H/4, W/4)
        with torch.no_grad():
            x_latent = self.vqgan.encoder(x)
            x_latent = self.vqgan.quant_conv(x_latent)
        return x_latent

    def decode(self, x_latent):
        # Decode Latent -> Ảnh
        with torch.no_grad():
            x_latent_quant, _, _ = self.vqgan.quantize(x_latent)
            out = self.vqgan.decode(x_latent_quant)
        return out

    def forward(self, x_gt, x_cond):
        """
        x_gt: Ảnh gốc (Ground Truth)
        x_cond: Ảnh điều kiện (Input thô từ Phase 1)
        """
        # 1. Chuyển sang Latent Space
        x1 = self.encode(x_gt).detach()   # Target data (t=1)
        xc = self.encode(x_cond).detach() # Condition (Source)
        
        b, c, h, w = x1.shape
        
        # 2. Optimal Transport Path (Interpolation)
        x0 = torch.randn_like(x1) # Noise Gaussian (t=0)
        t = torch.rand(b).to(x1.device) # Thời gian ngẫu nhiên [0, 1]
        
        # Reshape t để nhân broadcasing
        t_view = t.view(b, 1, 1, 1)
        
        # Công thức: x_t = (1 - t) * x0 + t * x1
        xt = (1 - t_view) * x0 + t_view * x1
        
        # Vector vận tốc mục tiêu: u_t = x1 - x0 (đạo hàm theo t của x_t)
        ut = x1 - x0
        
        # 3. Predict Velocity
        # Model nhận đầu vào là nối channel của (xt, xc)
        model_input = torch.cat([xt, xc], dim=1)
        vt = self.net(model_input, t)
        
        # 4. Loss MSE
        loss = F.mse_loss(vt, ut)
        return loss

    @torch.no_grad()
    def sample_infer(self, x_cond, steps=50):
        self.eval()
        # Encode điều kiện
        xc = self.encode(x_cond)
        b, c, h, w = xc.shape
        
        # Bắt đầu từ Noise thuần khiết
        x = torch.randn(b, c, h, w).to(xc.device)
        
        # Euler ODE Solver (Giải phương trình vi phân để đi từ Noise -> Data)
        dt = 1.0 / steps
        for i in range(steps):
            t_val = i / steps
            t = torch.ones(b).to(xc.device) * t_val
            
            model_input = torch.cat([x, xc], dim=1)
            v = self.net(model_input, t)
            
            # Cập nhật vị trí: x_{t+1} = x_t + v * dt
            x = x + v * dt
            
        # Decode latent cuối cùng ra ảnh
        img_out = self.decode(x)
        
        # Clip giá trị về [-1, 1] cho an toàn
        img_out = torch.clamp(img_out, -1, 1)
        return img_out