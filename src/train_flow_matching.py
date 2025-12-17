import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time
import cv2
from torchvision import transforms

# Imports
from model.flow_matching import LatentFlowMatchingModel
import options.options as option
import utils.utils as utils

# --- HELPER FUNCTIONS (FIXED) ---
def tensor2img(tensor):
    """
    Chuyển đổi Tensor (C, H, W) range [-1, 1] sang ảnh Numpy (H, W, C) uint8
    """
    # 1. Detach & Move to CPU
    tensor = tensor.detach().cpu()
    
    # 2. Denormalize: [-1, 1] -> [0, 1]
    tensor = (tensor + 1) / 2.0
    tensor = tensor.clamp(0, 1)
    
    # 3. Chuyển trục: (C, H, W) -> (H, W, C)
    if tensor.dim() == 3:
        img_np = tensor.permute(1, 2, 0).numpy()
    elif tensor.dim() == 4: # Nếu lỡ batch size = 1
        img_np = tensor.squeeze(0).permute(1, 2, 0).numpy()
    else:
        img_np = tensor.numpy()
        
    # 4. Scale lên 255
    img_np = (img_np * 255).astype(np.uint8)
    return img_np

# --- DATASET DEFINITION ---
class FlowMatchingDataset(Dataset):
    def __init__(self, data_root, size=256):
        self.pred_dir = os.path.join(data_root, 'false_pred') 
        self.label_dir = os.path.join(data_root, 'label')
        self.files = [f for f in os.listdir(self.pred_dir) if f.endswith('.npy')]
        self.size = size
        
        # Color map (4 classes)
        self.color_map = [
            [255, 255, 255], # 0
            [0, 128, 0],     # 1
            [255, 0, 0],     # 2
            [0, 0, 255]      # 3
        ]
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def index2rgb(self, map_idx):
        if len(map_idx.shape) == 3:
            if map_idx.shape[-1] > 1: 
                map_idx = np.argmax(map_idx, axis=-1)
            else:
                map_idx = map_idx.squeeze(-1)
        
        h, w = map_idx.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        for idx, color in enumerate(self.color_map):
            rgb[map_idx == idx] = color
            
        return rgb

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        fid = fname.split('.')[0]
        
        # Load Condition
        pred_map = np.load(os.path.join(self.pred_dir, fname))
        cond_img = self.index2rgb(pred_map)
        cond_img = cv2.resize(cond_img, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        
        # Load GT
        label_path = os.path.join(self.label_dir, f"{fid}.png")
        if os.path.exists(label_path):
            gt_img = cv2.imread(label_path)
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            gt_img = cv2.resize(gt_img, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        else:
            gt_img = np.zeros_like(cond_img)

        return {'GT': self.transform(gt_img), 'LQ': self.transform(cond_img)}

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='src/options/FlowMatching.yml')
    parser.add_argument('--data_dir', type=str, required=True, help='Path output gen data')
    args = parser.parse_args()
    
    opt = option.parse(args.opt, root='.')
    opt = option.dict_to_nonedict(opt)
    device = torch.device('cuda')
    
    dataset = FlowMatchingDataset(args.data_dir, size=256)
    loader = DataLoader(dataset, batch_size=opt['train']['batch_size'], shuffle=True, num_workers=2)
    
    print(f"[INFO] Training on {len(dataset)} samples for {opt['train']['total_steps']} steps.")

    model = LatentFlowMatchingModel(opt).to(device)
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=opt['train']['lr'])
    
    model.train()
    step = 0
    t0 = time.time()
    pbar = tqdm(total=opt['train']['total_steps'])
    
    while step < opt['train']['total_steps']:
        for batch in loader:
            if step >= opt['train']['total_steps']: break
            
            x_gt = batch['GT'].to(device)
            x_cond = batch['LQ'].to(device)
            
            optimizer.zero_grad()
            loss = model(x_gt, x_cond)
            loss.backward()
            optimizer.step()
            
            if step % opt['train']['log_freq'] == 0:
                print(f"Loss: {loss.item():.4f}")
            
            step += 1
            pbar.update(1)
            
    print(f"[INFO] Done in {(time.time()-t0)/60:.1f} mins.")
    
    # Test Visualization (Sử dụng hàm tensor2img nội bộ)
    print("[INFO] Visualizing results...")
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        n_vis = min(3, len(batch['GT']))
        x_cond = batch['LQ'][:n_vis].to(device)
        x_gt = batch['GT'][:n_vis].to(device)
        
        output = model.sample_infer(x_cond, steps=25)
        
        os.makedirs('results_fm', exist_ok=True)
        plt.figure(figsize=(12, 4 * n_vis))
        for i in range(n_vis):
            # Dùng hàm tensor2img đã định nghĩa ở trên thay vì utils.tensor2img
            plt.subplot(n_vis, 3, i*3 + 1)
            plt.imshow(tensor2img(x_cond[i]))
            plt.title("Input (Phase 1)")
            plt.axis('off')
            
            plt.subplot(n_vis, 3, i*3 + 2)
            plt.imshow(tensor2img(output[i]))
            plt.title("Latent FM Refined")
            plt.axis('off')
            
            plt.subplot(n_vis, 3, i*3 + 3)
            plt.imshow(tensor2img(x_gt[i]))
            plt.title("Ground Truth")
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig('results_fm/latent_check.png')
        print("[SUCCESS] Check results_fm/latent_check.png")

if __name__ == '__main__':
    main()