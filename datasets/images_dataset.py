import math
import random

import cv2
import imgaug.augmenters as iaa
import numpy as np
import skimage
import torchvision.transforms as transforms
from datasets.augmentation import BilinearResize
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from utils import data_utils


def brush_stroke_mask(img, color=(255,255,255)):
    # input :image,   code from: GPEN
    min_num_vertex = 5
    max_num_vertex = 8 #[8,10]
    mean_angle = 2*math.pi / 5
    angle_range = 2*math.pi / 15
    min_width = 12
    max_width = 80
    def generate_mask(H, W, img=None):
        average_radius = math.sqrt(H*H+W*W) / 20
        mask = Image.new('RGB', (W, H), 0)
        if img is not None: mask = img #Image.fromarray(img)
        # for _ in range(np.random.randint(1, 2)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)#[2*pi/5-2*pi/15]  4/15
        angle_max = mean_angle + np.random.uniform(0, angle_range)#[2*pi/5+2*pi/15] 8/15
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=color, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                         fill=color)

        return mask

    width, height = img.size
    mask = generate_mask(height, width, img)
    return mask

class UDADataset(Dataset):
	def __init__(self, source_root, target_root, opts,train=True, target_transform=None, source_transform=None):
		self.train=train
		if self.train:
			self.source_paths = sorted(data_utils.make_dataset(source_root + '/train/src'))
			self.target_paths = sorted(data_utils.make_dataset(target_root + '/train/trg'))
		else:
			self.source_paths = sorted(data_utils.make_dataset(source_root + '/test'))
			self.target_paths = sorted(data_utils.make_dataset(target_root + '/test'))
		self.source_transform = source_transform
		self.target_transform = target_transform
		if 'car' in opts.dataset_type:
			self.SR_transform = transforms.Compose([
				transforms.Resize((192, 256)),
				BilinearResize(factors=2),
				transforms.Resize((192, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		else:
			self.SR_transform=transforms.Compose([
					transforms.Resize((256, 256)),
					BilinearResize(factors=2),
					transforms.Resize((256, 256)),
					transforms.ToTensor(),
					transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		self.opts = opts

		self.rain=iaa.RainLayer(density=(0.03, 0.14),density_uniformity=(0.8, 1.0),drop_size=(0.01, 0.02),
		    drop_size_uniformity=(0.2, 0.5),angle=(-15, 15),speed=(0.05, 0.20),blur_sigma_fraction=(0.001, 0.001),)
	def __len__(self):
		return min(len(self.source_paths),len(self.target_paths))

	def __getitem__(self, index):
		#from :source ,to: target
		from_path = self.source_paths[index]
		src_im = Image.open(from_path)
		src_im = src_im.convert('RGB')

		to_path = self.target_paths[index%len(self.target_paths)]
		trg_im = Image.open(to_path).convert('RGB')
		temp=np.random.randint(3)
		if temp == 0:
			img_t_aug = self.rain(image=np.array(trg_im))
			trg_im = Image.fromarray(np.uint8(img_t_aug))
			trg_im = self.target_transform(trg_im)
			src_im = self.source_transform(src_im)
		elif temp==1:
			trg_im=brush_stroke_mask(trg_im)
			trg_im = trg_im.convert('RGB')
			trg_im = self.target_transform(trg_im)
			src_im = self.source_transform(src_im)
		elif temp==2:
			trg_im = self.SR_transform(trg_im)
			src_im = self.source_transform(src_im)

		return src_im, trg_im

class ADataset(Dataset):
	def __init__(self, source_root, target_root, opts,train=True, target_transform=None, source_transform=None):
		self.train=train
		if self.train:
			self.source_paths = sorted(data_utils.make_dataset(source_root + '/train/src'))
			self.target_paths = sorted(data_utils.make_dataset(target_root + '/train/trg'))
		else:
			self.source_paths = sorted(data_utils.make_dataset(source_root + '/test'))
			self.target_paths = sorted(data_utils.make_dataset(target_root + '/test'))
		self.source_transform = source_transform
		self.target_transform = target_transform
		if 'car' in opts.dataset_type:
			self.SR_transform = transforms.Compose([
				transforms.Resize((192, 256)),
				BilinearResize(factors=2),
				transforms.Resize((192, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		else:
			self.SR_transform=transforms.Compose([
					transforms.Resize((256, 256)),
					BilinearResize(factors=2),
					transforms.Resize((256, 256)),
					transforms.ToTensor(),
					transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		self.opts = opts
		self.rain=iaa.RainLayer(density=(0.03, 0.14),density_uniformity=(0.8, 1.0),drop_size=(0.01, 0.02),
		    drop_size_uniformity=(0.2, 0.5),angle=(-15, 15),speed=(0.05, 0.20),blur_sigma_fraction=(0.001, 0.001),)
	def __len__(self):
		return max(len(self.source_paths),len(self.target_paths))

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		src_im = Image.open(from_path).convert('RGB')

		to_path = self.target_paths[index]
		trg_im = Image.open(to_path).convert('RGB')
		trg_gt_img=Image.open(to_path).convert('RGB')
		temp=np.random.randint(3)
		if temp == 0:
			img_t_aug = self.rain(image=np.array(trg_im))
			trg_im = Image.fromarray(np.uint8(img_t_aug))
			trg_im = self.target_transform(trg_im)
			src_im = self.source_transform(src_im)
		elif temp==1:
			trg_im=brush_stroke_mask(trg_im)
			trg_im = trg_im.convert('RGB')
			trg_im = self.target_transform(trg_im)
			src_im = self.source_transform(src_im)
		elif temp==2:
			trg_im = self.SR_transform(trg_im)
			src_im = self.source_transform(src_im)

		trg_gt_img=self.source_transform(trg_gt_img)

		return src_im, trg_im,trg_gt_img
class ImagesDataset(Dataset):

	def __init__(self, source_root, target_root, opts,target_transform=None, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		#from :source to target
		from_path = self.source_paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB')

		to_path = self.target_paths[index]
		to_im = Image.open(to_path).convert('RGB')
		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im

		return from_im, to_im
