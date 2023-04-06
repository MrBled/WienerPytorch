import torch
import torchvision
from torch import nn
from torchvision.utils import save_image
import torch.nn.functional as F


class Wiener_3D(nn.Module):
    def __init__(self, weight_fft=0.3, weight_interp=0.3, default_size=32):
        super(Wiener_3D, self).__init__()
        self.relu = nn.ReLU()
        self.weight_fft = weight_fft
        self.weight_interp = weight_interp
        self.default_size = default_size


    def forward(self, I, noise_std):
        block_size = self.default_size
        B = I.shape[0]
        height = I.shape[2] + block_size * 2
        width = I.shape[3] + block_size * 2
        IR = torch.zeros(height, width, dtype=torch.float64)

        bx = block_size
        hbx = bx / 2
        overlap = 4
        # CALCULATE THE COSINE WINDOW
        win1x = torch.exp(-(torch.arange(-hbx + 0.5, hbx - 0.5 + 1)) ** 2 / (self.weight_fft * hbx * hbx)).cuda() # 0.3 default
        win1x = win1x.unsqueeze(0).repeat(block_size, 1)
        win1y = torch.exp(-(torch.arange(-hbx + 0.5, hbx - 0.5 + 1)) ** 2 / (self.weight_fft * hbx * hbx)).cuda()
        win1y = win1y.unsqueeze(1).repeat(1, block_size)
        
        win = win1y * win1x
        win = win.unsqueeze(0).repeat(3, 1, 1) # Expand for colour rgb

        win1x = torch.exp(-(torch.arange(-hbx + 0.5, hbx - 0.5 + 1)) ** 2 / (self.weight_interp * hbx * hbx)).cuda() #new 0.2 default
        win1x = win1x.unsqueeze(0).repeat(block_size, 1)
        win1y = torch.exp(-(torch.arange(-hbx + 0.5, hbx - 0.5 + 1)) ** 2 / (self.weight_interp * hbx * hbx)).cuda()
        win1y = win1y.unsqueeze(1).repeat(1, block_size)
        win_interp = win1y * win1x
        win_interp = win_interp.unsqueeze(0).repeat(3, 1, 1) # Expand for colour rgb

        # CREATE THE UNFOLDED IMAGE SLICES
        kernel_size = bx
        stride = int(bx / overlap)
        I2 = F.pad(input=I, pad=(block_size, block_size, block_size, block_size), mode="reflect")
        noise_std = F.pad(input=noise_std, pad=(block_size, block_size, block_size, block_size), mode="reflect")

        patches2 = I2.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)

        IR_mask = torch.ones(height, width).cuda()
        IR_mask = IR_mask.unsqueeze(0).unsqueeze(0)
        IR_mask = IR_mask.repeat(B, 3, 1, 1)
        IR_mask = IR_mask.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
        std_patches = noise_std.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
        std_patches = torch.mean(std_patches, (4, 5), keepdim=True)
        std_patches = std_patches.repeat(1, 1, 1, 1, self.default_size, self.default_size)
        # NORMALISE AND WINDOW
        window_patches = win.unsqueeze(1).unsqueeze(2).unsqueeze(0).repeat(B, 1, patches2.size(2), patches2.size(3), 1, 1)
        win_interp = win_interp.unsqueeze(1).unsqueeze(2).unsqueeze(0).repeat(B, 1, patches2.size(2), patches2.size(3), 1, 1)

        mean_patches = torch.mean(patches2, (4,5), keepdim=True)

        mean_patches = mean_patches.repeat(1, 1, 1, 1, block_size, block_size)

        zero_mean = patches2 - mean_patches
        windowed_patches = zero_mean * window_patches
        IR_mask = IR_mask * win_interp * window_patches
        # APPLY THE WIENER FILTER

        freq_transf = torch.fft.fftn(windowed_patches, dim=(1, 4, 5))
        Pss = torch.abs(freq_transf)**2
        eps = 1e-15
        Pss = Pss + eps

        # Normalise, window, get wiener ########
        Pvv = torch.mean(torch.pow(win, 2), (1, 2), keepdim=True) * torch.numel(win[0])

        Pvv = Pvv.unsqueeze(1).unsqueeze(1).unsqueeze(0).repeat(std_patches.size(0), 1, std_patches.size(2), std_patches.size(3), self.default_size, self.default_size)
        Pvv = Pvv * (std_patches**2)
        # WIENER CORING #########################
        H = Pss - Pvv
        H = self.relu(H)
        H = H / Pss
        filt_freq_block = H * freq_transf

        filt_data_block = torch.fft.ifftn(filt_freq_block, dim=(1, 4, 5)).real

        filt_data_block = (filt_data_block + mean_patches * window_patches) 
        filt_data_block = filt_data_block * win_interp                      
        # REASSEMBLE IMAGE
        patches = filt_data_block.contiguous().view(B, 3, -1, kernel_size * kernel_size)
        patches = patches.permute(0, 1, 3, 2)
        patches = patches.contiguous().view(B, 3 * kernel_size * kernel_size, -1)

        IR_mask = IR_mask.contiguous().view(B, 3, -1, kernel_size * kernel_size)
        IR_mask = IR_mask.permute(0, 1, 3, 2)
        IR_mask = IR_mask.contiguous().view(B, 3 * kernel_size * kernel_size, -1)

        IR = F.fold(patches, output_size=(height, width), kernel_size=kernel_size, stride=stride)
        IR_mask = F.fold(IR_mask, output_size=(height, width), kernel_size=kernel_size, stride=stride)

        # NORMALIZE IR
        IR = IR + eps
        IR_mask = IR_mask + eps
        IR = IR / IR_mask
        IR = IR[:, :, block_size:I.shape[2] + block_size, block_size:I.shape[3] + block_size]
        return IR
