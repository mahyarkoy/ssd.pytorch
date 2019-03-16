"""AOD Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Mahyar
"""
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

class AODDetection(data.Dataset):
	def __init__(self, im_names, bboxes,
				 transform=None):
		
		self.transform = transform
		self.imn_list = im_names
		self.bbox_list = bboxes
		self.name = 'AOD'

	def __getitem__(self, index):
		im, gt, h, w = self.pull_item(index)

		return im, gt

	def __len__(self):
		return len(self.imn_list)

	def pull_item(self, index):
		### read image from using the name in imn_list
		im_name = self.imn_list[index]
		img = cv2.imread(im_name)
		h, w, c = img.shape

		### read bounding boxes and scale: [xmin, ymin, xmax, ymax]
		im_bbox = 1. * np.array(self.bbox_list[index]) / np.array([w, h, w, h])

		### add a dummy label to the end of each bbox: [xmin, ymin, xmax, ymax, 1]
		im_lab = 0 * np.ones((im_bbox.shape[0], 1), dtype=np.int)
		target = np.concatenate((im_bbox, im_lab), axis=1)

		### apply image transformation
		if self.transform is not None:
			img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
			# to rgb
			img = img[:, :, (2, 1, 0)]
			target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
		### transpose image to channel first: [B, C, H, W]
		return torch.from_numpy(img).permute(2, 0, 1), target, h, w

	def pull_image(self, index):
		### read image from using the name in imn_list
		im_name = self.imn_list[index]
		return cv2.imread(im_name, cv2.IMREAD_COLOR)

	def pull_anno(self, index):
		'''Returns the original annotation of image at index

		Note: not using self.__getitem__(), as any transformations passed in
		could mess up this functionality.

		Argument:
			index (int): index of img to get annotation of
		Return:
			im_name, [[bbox coords, class_id],...]
		'''
		### read image name from imn_list
		im_name = self.imn_list[index]

		### read bounding boxes and do not scale: [xmin, ymin, xmax, ymax]
		im_bbox = 1. * np.array([self.bbox_list[index]])

		### add a dummy label to the end of each bbox: [xmin, ymin, xmax, ymax, 1]
		im_lab = 0 * np.ones((im_bbox.shape[0], 1), dtype=np.int)
		target = np.concatenate((im_bbox, im_lab), axis=1)

		return im_name, target.tolist()

	def pull_tensor(self, index):
		'''Returns the original image at an index in tensor form

		Note: not using self.__getitem__(), as any transformations passed in
		could mess up this functionality.

		Argument:
			index (int): index of img to show
		Return:
			tensorized version of img, squeezed
		'''
		return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
