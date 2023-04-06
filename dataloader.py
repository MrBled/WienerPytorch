import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import glob
from torch.utils.data import Dataset
import cv2

def read_img(filename):
    img = cv2.imread(filename)
    if img is None or isinstance(img, str):
        print("invalid img")
        print(filename)
        return "None"
    if img.ndim < 3:
        print("single dim img")
        img = np.expand_dims(img, 2)
        img = img[:,:,::-1] / 255.0
    else:
        img = img[:,:,::-1] / 255.0

    img = np.array(img).astype('float32')

    return img


class load_noise_imgs(Dataset):
    def __init__(self, root_dir, sample_num, patch_size=500):
        self.patch_size = patch_size
        folders = glob.glob(root_dir + '/*')
        folders.sort()
        # GET ALL FOLDERS IN DIR
        self.im_fns = [None] * sample_num
        for i in range(sample_num):
            self.im_fns[i] = []
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.CenterCrop((patch_size, patch_size)),
            ])
        # GET ALL FILES IN ALL FOLDERS ITERATIVELY
        for ind, folder in enumerate(folders):
            imgs = glob.glob(folder + '/*.png*')
            imgs.sort()

            for ind, image in enumerate(imgs):
                self.im_fns[ind % sample_num].append(image) # EACH image takes a spot and lists all patches eg 420 images, 49 patches in each

    def __len__(self):
        l = len(self.im_fns)
        return l

    def __getitem__(self, idx):
        curr_fn = self.im_fns[idx][0]
        curr_img = read_img(curr_fn)
        curr_img = self.transforms(curr_img)
        return curr_img