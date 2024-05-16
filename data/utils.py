from torchvision import transforms
import numpy as np
from . import TRANSFORMS
from torchvision.transforms.v2 import Compose, Transform, RandomAutocontrast, RandomAffine, CenterCrop, RandomEqualize,RandomPhotometricDistort, RandomChoice, RandomAdjustSharpness, RandomHorizontalFlip, Resize, RandomVerticalFlip, ColorJitter, RandomApply, GaussianBlur, RandomPosterize, Normalize


def get_v2_transforms(mode):
	if mode=="train":
		tranform = Compose(
      [
          Resize((256, 256), antialias=True),
          #CenterCrop(size=(224, 224)),
          RandomAdjustSharpness(sharpness_factor=4, p=0.5),
          RandomApply(torch.nn.ModuleList([
              RandomChoice([
                  GaussianBlur(kernel_size = 5),
                  GaussianBlur(kernel_size = 7),
                  GaussianBlur(kernel_size = 9),
              ])
            ]), p=0.5),
          RandomPhotometricDistort(p=0.5),
          RandomApply(torch.nn.ModuleList([
            ColorJitter(brightness=1.0,
                        contrast = 0.5,
                        saturation = 0.5,
                        hue=0.3
                        )
            ]), p=0.5),
          #RandomEqualize(p=0.2),
          RandomAutocontrast(p=0.2),
          RandomHorizontalFlip(p=0.5),
          RandomAffine(degrees=45, translate=(0.4, 0.4), scale=(0.7, 1.1)),
          RandomVerticalFlip(p=0.5),
          Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

      ],
  )
  if mode == "test":
	transform = Compose(
      [
          Resize((256, 256), antialias=True),
          #CenterCrop(size=(224, 224)),
          RandomApply(torch.nn.ModuleList([
              RandomChoice([
                  GaussianBlur(kernel_size = 5),
                  GaussianBlur(kernel_size = 7),
                  GaussianBlur(kernel_size = 9),
              ])
            ]), p=0.5),
          #RandomPosterize( bits=3, p=0.5),
          RandomPhotometricDistort(p=0.3),
          RandomApply(torch.nn.ModuleList([
            ColorJitter(brightness=1.0,
                        contrast = 0.5,
                        saturation = 0.5,
                        hue=0.3
                        )
            ]), p=0.5),
          #RandomEqualize(p=0.3),
          RandomAutocontrast(p=0.3),
          RandomHorizontalFlip(p=0.5),
          RandomAffine(degrees=45, translate=(0.4, 0.4), scale=(0.7, 1.1)),
          RandomVerticalFlip(p=0.5),
          Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ]
  )
  return transform


def get_transforms(cfg, train, cfg_transforms):
	transform_list = []
	for t in cfg_transforms:
		t = {k: v for k, v in t.items()}
		t_type = t.pop('type')
		t_tran = TRANSFORMS.get_module(t_type)(**t)
		transform_list.extend(t_tran) if isinstance(t_tran, list) else transform_list.append(t_tran)
	transform_out = TRANSFORMS.get_module('Compose')(transform_list)

	# if train:
	# 	if cfg.size <= 32:
	# 		transform_out[0] = transforms.RandomCrop(cfg.size, padding=4)
	return transform_out


def make_divisible(v, divisor=8, min_value=None):
	if min_value is None:
		min_value = divisor
	new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
	# Make sure that round down does not go down by more than 10%.
	if new_v < 0.9 * v:
		new_v += divisor
	return new_v
