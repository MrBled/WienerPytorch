import torch
from torchvision.utils import save_image
from torch.nn import functional as F
# from torch import autograd
from wiener import Wiener_3D
from torch import nn
from dataloader import load_noise_imgs

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    dataset_path = "/home/bledc/dataset/wiener_demo_gaussian_imgs"
    test_dataset = load_noise_imgs(dataset_path, 100)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1,
        shuffle=False, num_workers=8,
        pin_memory=True, drop_last=False)

    print(device)

    std = 35
    win_size=32
    overlap=4
    wiener_denoiser = nn.DataParallel(Wiener_3D(0.3, 0.6, win_size, overlap)).cuda()
    for i, noise_img in enumerate(test_loader):
        noise_img = noise_img.cuda()

        
        k = 7.8 
        std_curr = torch.full((noise_img.size(0), noise_img.size(1), noise_img.size(2), noise_img.size(3)), std / 255)
        
        wiener_filtered = wiener_denoiser(noise_img, std_curr * k)
        wiener_filtered = torch.clamp(wiener_filtered, 0, 1)
            
        save_image(wiener_filtered, f"{i}_denoise.png")
        save_image(noise_img, f"{i}_noisy.png")
        print(f"Image {i} denoised.")
