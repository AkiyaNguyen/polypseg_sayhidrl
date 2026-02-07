import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import shutil
import torch.nn.functional as F
from torch.utils.data import random_split

def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def binary_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')
    
def add_slash(paths: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(path if path.endswith('/') else path + '/' for path in paths)

## =========== utils functions for loading data ===========

def gt_to_normalized_numpy(gt: Image.Image) -> np.ndarray:
    gt_array = np.asarray(gt, np.float32)
    gt_array /= (gt_array.max() + 1e-8)
    return gt_array

def process_output_for_inference(P1, P2, gt: np.ndarray):
    res = F.upsample(P1+P2, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    return res

## ===========helper function for inference===========

class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, trainsize, augmentations):
        image_root, gt_root = add_slash((image_root, gt_root))
        self.trainsize = trainsize
        self.augmentations = augmentations
        print(self.augmentations)
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        if self.augmentations == 'True':
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, interpolation=transforms.InterpolationMode.NEAREST, expand=False, center=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, interpolation=transforms.InterpolationMode.NEAREST, expand=False, center=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            
        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            

    def __getitem__(self, index):
        
        image = rgb_loader(self.images[index])
        gt = binary_loader(self.gts[index])
        
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)
            
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.Resampling.BILINEAR), gt.resize((w, h), Image.Resampling.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True, augmentation=False):

    dataset = PolypDataset(image_root, gt_root, trainsize, augmentation)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


def get_train_val_loader(image_root, gt_root, batchsize, trainsize, train_shuffle=True, num_workers=4, pin_memory=True, augmentation=False, validation_ratio: float=0.2):
    """
    argument: shuffle: decide if the train_loader should be shuffle
    """    
    dataset = PolypDataset(image_root, gt_root, trainsize, augmentation)
    
    num_valid = max(1, int(len(dataset) * validation_ratio))
    num_train = len(dataset) - num_valid
    train_set, val_set = random_split(dataset, [num_train, num_valid])

    train_loader = data.DataLoader(dataset=train_set,
                                    batch_size=batchsize,
                                    shuffle=train_shuffle,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory)
    val_loader = data.DataLoader(dataset=val_set,
                                    batch_size=batchsize,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory)
    
    return train_loader, val_loader


# def get_train_val_loader(image_root, gt_root, batchsi)

class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        image_root, gt_root = add_slash((image_root, gt_root))
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0) #type: ignore
        gt = binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name
class dummy_test_dataset(test_dataset):
    base_dataset_path = '../dataset/TestDataset' ## relative path
    
    def __init__(self, img_path_list, gt_path_list, testsize, test_dataset_name: str):
        os.makedirs(os.path.join(self.base_dataset_path, test_dataset_name), exist_ok=True)
        os.makedirs(os.path.join(self.base_dataset_path, test_dataset_name, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.base_dataset_path, test_dataset_name, 'masks'), exist_ok=True)
        for img_path, gt_path in zip(img_path_list, gt_path_list):
            shutil.copy(img_path, os.path.join(self.base_dataset_path, test_dataset_name, 'images', os.path.basename(img_path)))
            shutil.copy(gt_path, os.path.join(self.base_dataset_path, test_dataset_name, 'masks', os.path.basename(gt_path)))

        image_root = os.path.join(self.base_dataset_path, test_dataset_name, 'images') + '/'
        gt_root = os.path.join(self.base_dataset_path, test_dataset_name, 'masks') + '/'
        super().__init__(image_root, gt_root, testsize)

## =========== original dataloader for polyp segmentation tasks ===========

class DepthAugmentPolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks with depth augmentation
    """
    def __init__(self, image_root, depth_root, gt_root, trainsize, augmentations):
        image_root, gt_root, depth_root = add_slash((image_root, gt_root, depth_root))
        self.trainsize = trainsize
        self.augmentations = augmentations
        print(self.augmentations)
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        
        self.images = sorted(self.images)
        self.depths = sorted(self.depths)
        self.gts = sorted(self.gts)

        self.filter_files()
        self.size = len(self.images)
        # The rest of the implementation would be similar to PolypDataset,
        # but would include loading and transforming depth maps as well.
        # For brevity, the full implementation is not included here.
        if self.augmentations == 'True':
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, interpolation=transforms.InterpolationMode.NEAREST, expand=False, center=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, interpolation=transforms.InterpolationMode.NEAREST, expand=False, center=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            
        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            

    def filter_files(self):
        print("Filtering files to ensure matching image, depth, and gt sizes.")
        print(len(self.images), len(self.depths), len(self.gts))
        assert len(self.images) == len(self.gts) == len(self.depths)
        images = []
        depths = []
        gts = []
        for img_path, depth_path, gt_path in zip(self.images, self.depths, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                depths.append(depth_path)
                gts.append(gt_path)
        self.images = images
        self.depths = depths
        self.gts = gts
    def __getitem__(self, index):
        image = rgb_loader(self.images[index])
        depth = rgb_loader(self.depths[index])
        gt = binary_loader(self.gts[index])
        # Apply transformations similar to PolypDataset
        # including depth map transformations.
        # For brevity, the full implementation is not included here
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)
            depth = self.img_transform(depth)
            
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        
        return (image, depth), gt
    
    # def resize(self, img, gt):
    #     assert img.size == gt.size
    #     w, h = img.size
    #     if h < self.trainsize or w < self.trainsize:
    #         h = max(h, self.trainsize)
    #         w = max(w, self.trainsize)
    #         return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
    #     else:
    #         return img, gt

    def __len__(self):
        return self.size
        
def get_depth_augment_loader(image_root, depth_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True, augmentation=False):

    dataset = DepthAugmentPolypDataset(image_root, depth_root, gt_root, trainsize, augmentation)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

def get_depth_train_val_loader(image_root, depth_root, gt_root, batchsize, trainsize, train_shuffle=True, num_workers=4, pin_memory=True, augmentation=False, validation_ratio: float=0.2):
    """
    argument: shuffle: decide if the train_loader should be shuffle
    """    
    dataset = DepthAugmentPolypDataset(image_root, depth_root, gt_root, trainsize, augmentation)
    
    num_valid = max(1, int(len(dataset) * validation_ratio))
    num_train = len(dataset) - num_valid
    train_set, val_set = random_split(dataset, [num_train, num_valid])

    train_loader = data.DataLoader(dataset=train_set,
                                    batch_size=batchsize,
                                    shuffle=train_shuffle,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory)
    val_loader = data.DataLoader(dataset=val_set,
                                    batch_size=batchsize,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory)
    
    return train_loader, val_loader


class test_depth_enhance_dataset:
    def __init__(self, image_root, gt_root, depth_root, testsize):
        image_root, gt_root, depth_root = add_slash((image_root, gt_root, depth_root))
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0) #type: ignore

        depth = rgb_loader(self.depths[self.index])
        depth = self.transform(depth).unsqueeze(0) #type: ignore

        gt = binary_loader(self.gts[self.index])

        name = self.images[self.index].split('/')[-1]

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        self.index += 1
        return image, depth, gt, name

class dummy_test_depth_enhance_dataset(test_depth_enhance_dataset):
    base_dataset_path = '../dataset/TestDataset' ## relative path
    def __init__(self, img_path_list, gt_path_list, depth_path_list, testsize, test_dataset_name: str):
        os.makedirs(os.path.join(self.base_dataset_path, test_dataset_name), exist_ok=True)
        os.makedirs(os.path.join(self.base_dataset_path, test_dataset_name, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.base_dataset_path, test_dataset_name, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(self.base_dataset_path, test_dataset_name, 'depths'), exist_ok=True)
        for img_path, gt_path, depth_path in zip(img_path_list, gt_path_list, depth_path_list):
            shutil.copy(img_path, os.path.join(self.base_dataset_path, test_dataset_name, 'images', os.path.basename(img_path)))
            shutil.copy(gt_path, os.path.join(self.base_dataset_path, test_dataset_name, 'masks', os.path.basename(gt_path)))
            shutil.copy(depth_path, os.path.join(self.base_dataset_path, test_dataset_name, 'depths', os.path.basename(depth_path)))
        image_root = os.path.join(self.base_dataset_path, test_dataset_name, 'images') + '/'
        gt_root = os.path.join(self.base_dataset_path, test_dataset_name, 'masks') + '/'
        depth_root = os.path.join(self.base_dataset_path, test_dataset_name, 'depths') + '/'
        super().__init__(image_root, gt_root, depth_root, testsize)

## =========== dataloader for polyp segmentation tasks with depth augmentation ===========

if __name__ == '__main__':
    # test_dataset = test_dataset(image_root='../dataset/TestDataset/Kvasir/images/', 
    #                             gt_root='../dataset/TestDataset/Kvasir/masks/', testsize=512)
    # image, gt, name = test_dataset.load_data()
    # print("shape and type of image: ", image.shape, image.dtype)
    # print("shape and type of gt: ", gt.size, type(gt))
    # print("gt mode: ", gt.mode)
    # ## expected: image of shape (1, 3, H, W) and dtype torch.float32
    # ## expected: gt of size (H, W) and mode 'L' and type PIL.Image.Image

    ## ===== test dummy_test_dataset and dummy_test_depth_enhance_dataset =====
    image_path_list = ['../dataset/TestDataset/CVC-ColonDB/images/2.png', '../dataset/TestDataset/CVC-ColonDB/images/3.png']
    gt_path_list = ['../dataset/TestDataset/CVC-ColonDB/masks/2.png', '../dataset/TestDataset/CVC-ColonDB/masks/3.png']
    # depth_path_list = ['../dataset/TestDataset/CVC-ColonDB/depths/2.png', '../dataset/TestDataset/CVC-ColonDB/depths/3.png']
    test_dataset_name = 'dummy'
    my_test_dataset = dummy_test_dataset(image_path_list, gt_path_list, testsize=512, test_dataset_name=test_dataset_name)
    image, gt, name = my_test_dataset.load_data()
    print("shape and type of image: ", image.shape, image.dtype)
    # print("shape and type of depth: ", depth.shape, depth.dtype)
    print("shape and type of gt: ", gt.size, type(gt))
    print("name: ", name)
    ## expected: image of shape (1, 3, H, W) and dtype torch.float32
    ## expected: gt of size (H, W) and mode 'L' and type PIL.Image.Image
    ## expected: name of type string