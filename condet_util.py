import numpy as np
import os
from progressbar import ETA, Bar, Percentage, ProgressBar
import pickle as pk
import glob
from PIL import Image
import xml.etree.ElementTree as xmlet

def read_image(im_path, im_size, sqcrop=True, bbox=None, verbose=False):
	im = Image.open(im_path)
	w, h = im.size
	if sqcrop:
		im_cut = min(w, h)
		left = (w - im_cut) //2
		top = (h - im_cut) //2
		right = (w + im_cut) //2
		bottom = (h + im_cut) //2
		im_sq = im.crop((left, top, right, bottom))
	elif bbox is not None:
		left = bbox[0]
		top = bbox[1]
		right = bbox[2]
		bottom = bbox[3]
		im_sq = im.crop((left, top, right, bottom))
	else:
		im_sq = im
	im_re_pil = im_sq.resize((im_size, im_size), Image.BILINEAR)
	im_re = np.array(im_re_pil.getdata())
	### next line is because pil removes the channels for black and white images!!!
	im_re = im_re if len(im_re.shape) > 1 else np.repeat(im_re[..., np.newaxis], 3, axis=1)
	im_re = im_re.reshape((im_size, im_size, 3))
	im.close()
	im_o = im_re / 128.0 - 1.0 
	return im_o if not verbose else (im_o, w, h)

def read_voc_files(path):
	sets_path = path + '/ImageSets/Main/'
	class_list = [
		'aeroplane', 'bicycle', 'bird', 'boat',
		'bottle', 'bus', 'car', 'cat', 'chair',
		'cow', 'diningtable', 'dog', 'horse',
		'motorbike', 'person', 'pottedplant',
		'sheep', 'sofa', 'train', 'tvmonitor']
	im_sets_train = list()
	im_sets_val = list()
	im_sets_test = list()
	### read image names in each set
	for i, c in enumerate(class_list):
		with open(sets_path+c+'_train.txt', 'r') as fs:
			im_sets_train.append([l.strip().split(' ')[0] for l in fs \
				if int(l.strip().split(' ')[-1]) == 1])
		with open(sets_path+c+'_val.txt', 'r') as fs:
			im_sets_val.append([l.strip().split(' ')[0] for l in fs \
				if int(l.strip().split(' ')[-1]) == 1])
		with open(sets_path+c+'_test.txt', 'r') as fs:
			im_sets_test.append([l.strip().split(' ')[0] for l in fs \
				if int(l.strip().split(' ')[-1]) == 1])
	return class_list, im_sets_train, im_sets_val, im_sets_test

'''
Returns a bbox matrix of shape (B, 4) where B is the number of objects with class_name.
path: xml file containing voc annotation.
'''
def read_voc_bbox(path, class_name, normalize=False):
	e = xmlet.parse(path).getroot()
	h = float(e.find('./size/height').text)
	w = float(e.find('./size/width').text)
	bbox = list()
	for obj in e.findall('object'):
		if obj.find('name').text == class_name:
			bb = map(int, 
				[obj.find('./bndbox/xmin').text, obj.find('./bndbox/ymin').text, 
				obj.find('./bndbox/xmax').text, obj.find('./bndbox/ymax').text])
			bb = list(bb)
			if normalize:
				bb = [bb[0]/w, bb[1]/h, bb[2]/w, bb[3]/h]
			bbox.append(bb)
	return np.array(bbox).reshape((-1, 4))

def prune_voc_hard(path, imnc_list, h_thr, w_thr):
	ann_path = path + '/Annotations/'
	keep_list = list()
	for i, (imn, c) in enumerate(imnc_list):
		bbox = read_voc_bbox(ann_path+imn+'.xml', c, normalize=True)
		bbox_w = bbox[:, 2] - bbox[:, 0]
		bbox_h = bbox[:, 3] - bbox[:, 1]
		if np.min(bbox_w) > w_thr and np.min(bbox_h) > h_thr:
			keep_list.append((imn, c))
	return keep_list

'''
Returns a list of (im_name, class_name) for each co, test, and train set.
co_num, test_num, train_num: list of size class_num, indicating num per class.
'''
def prep_voc(path, co_num, test_num, train_num, h_thr=0.2, w_thr=0.2):
	class_list, im_names_train, im_names_val, im_names_test = read_voc_files(path)
	co_list = list()
	test_list = list()
	train_list = list()
	for i, c in enumerate(class_list):
		if co_num[i] > 0 or train_num[i] > 0:
			imn_val = im_names_val[i]
			imn_train = im_names_train[i]
			imnc_trainval = [(imn, c) for imn in imn_val + imn_train]
			np.random.shuffle(imnc_trainval)
			imnc_trainval = prune_voc_hard(path, imnc_trainval, h_thr, w_thr)
			co_list.extend(imnc_trainval[:co_num[i]])
			train_list.extend(imnc_trainval[co_num[i]:co_num[i]+train_num[i]])
		
		if test_num[i] > 0:
			imn_test = im_names_test[i]
			imnc_test = [(imn, c) for imn in imn_test]
			np.random.shuffle(imnc_test)
			imnc_test = prune_voc_hard(path, imnc_test, h_thr, w_thr)
			test_list.extend(imnc_test[:test_num[i]])

	return co_list, test_list, train_list

'''
Prepare VOC data for AODDetection dataset.
voc_aod_path: location of pickle file storing condet voc preps.
'''
def prep_voc_aod_data(voc_aod_path, phase='train'):
	voc_dir = '/media/evl/Public/Mahyar/Data/voc/VOCdevkit/VOC2007/'
	print('>>> preparing data from: ', voc_aod_path)
	with open(voc_aod_path, 'rb') as fs:
		voc_co_list, voc_test_list, voc_train_list = pk.load(fs)
	voc_data_list = voc_test_list if phase == 'test' else voc_co_list
	im_names, bboxes = read_voc_ssd(voc_dir, voc_data_list)
	print('>>> im_names size: ', len(im_names))
	print('>>> bboxes size: ', len(bboxes))
	return im_names, bboxes

'''
Reads the (im_name, class_name) in data_list and constructs bboxes and im_names.
Used for constructing AOD dataset in ssd.
voc_dir: voc dataset directory.
data_list: list of (im_name, class_name) to be read from voc_dir.
'''
def read_voc_ssd(voc_dir, data_list):
	im_dir = voc_dir + '/JPEGImages/'
	ann_dir = voc_dir + '/Annotations/'
	imn_list = list()
	bbox_list = list()
	for imn, c in data_list:
		bbox_list.append(read_voc_bbox(ann_dir+imn+'.xml', c))
		imn_list.append(im_dir+imn+'.jpg')
	return imn_list, bbox_list

'''
Reading VOC dataset.
co_list, test_list, and train_list: list of (im_name, class_name) for each set.
'''
def read_voc(path, co_list, test_list, train_list, im_size=128, co_size=64):
	### inits
	co_num = len(co_list)
	test_num = len(test_list)
	train_num = len(train_list)
	im_path = path + '/JPEGImages/'
	ann_path = path + '/Annotations/'
	train_im = list()
	test_im = list()
	co_im = list()
	train_labs = list()
	test_labs = list()
	co_labs = list()
	train_bb = list()
	test_bb = list()
	total_num = co_num + test_num + train_num
	print('>>> Reading VOC from: '+ path)
	widgets = ["VOC", Percentage(), Bar(), ETA()]
	pbar = ProgressBar(maxval=total_num, widgets=widgets)
	pbar.start()
	counter = 0

	### read content images
	for imn, c in co_list:
		counter += 1
		pbar.update(counter)
		bbox = read_voc_bbox(ann_path+imn+'.xml', c)
		im = read_image(im_path+imn+'.jpg', co_size, sqcrop=False, bbox=bbox[0])
		co_im.append(im)
		co_labs.append(c)

	### read test images
	for imn, c in test_list:
		counter += 1
		pbar.update(counter)
		bbox = read_voc_bbox(ann_path+imn+'.xml', c, normalize=True)
		im, w, h = read_image(im_path+imn+'.jpg', im_size, 
			sqcrop=False, verbose=True)
		test_im.append(im)
		test_labs.append(c)
		test_bb.append((bbox * im_size).astype(int))

	### read train images
	for imn, c in train_list:
		counter += 1
		pbar.update(counter)
		bbox = read_voc_bbox(ann_path+imn+'.xml', c, normalize=True)
		im, w, h = read_image(im_path+imn+'.jpg', im_size, 
			sqcrop=False, verbose=True)
		train_im.append(im)
		train_labs.append(c)
		train_bb.append((bbox * im_size).astype(int))

	return (np.array(co_im), co_labs), \
		(np.array(test_im), test_bb, test_labs), \
		(np.array(train_im), train_bb, train_labs)

def read_cub_file(fname):
	vals = list()
	with open(fname, 'r') as fs:
		for l in fs:
			vals.append(l.strip().split(' ')[1:])
	return vals

'''
Prepare CUB data for AODDetection dataset.
cub_aod_path: location of pickle file storing condet cub preps.
'''
def prep_cub_aod_data(cub_aod_path, phase='train'):
	cub_path = '/media/evl/Public/Mahyar/Data/cub/CUB_200_2011'
	print('>>> preparing data from: ', cub_aod_path)
	with open(cub_aod_path, 'rb') as fs:
		cub_co_order, cub_test_order, cub_train_order = pk.load(fs, encoding='latin1')
	cub_data_order = cub_test_order if phase == 'test' else cub_co_order
	im_names, bboxes = read_cub_ssd(cub_path, cub_data_order)
	print('>>> im_names size: ', len(im_names))
	print('>>> bboxes size: ', len(bboxes))
	return im_names, bboxes

'''
Reads the indices in data_order and constructs bboxes and im_names.
Used for constructing AOD dataset in ssd.
cub_dir: cub dataset directory.
data_list: list of indices to be read from images.txt
'''
def read_cub_ssd(cub_dir, data_order):
	imn_list = list()
	bbox_list = list()
	im_fnames = np.array([v[0] for v in read_cub_file(cub_dir+'/images.txt')])
	im_class = np.array([int(v[0]) for v in read_cub_file(cub_dir+'/image_class_labels.txt')]) - 1
	im_bbox = [[int(float(v)) for v in bb] for bb in read_cub_file(cub_dir+'/bounding_boxes.txt')]
	for i in data_order:
		imn_list.append(cub_dir+'/images/'+im_fnames[i])
		bbox = im_bbox[i]
		bbox_list.append(np.array(
			[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]).reshape(1,4))
	
	return imn_list, bbox_list

'''
Reading CUB dataset.
co_order, test_order, train_order: the list of indices on images.txt for co, test, and train splits.
'''
def read_cub(cub_path, co_order, test_order, train_order, im_size=128, co_size=64):
	### read image file names, bboxes, and classes
	im_fnames = np.array([v[0] for v in read_cub_file(cub_path+'/images.txt')])
	im_class = np.array([int(v[0]) for v in read_cub_file(cub_path+'/image_class_labels.txt')]) - 1
	im_bbox = np.array(
		[[int(float(v)) for v in bb] for bb in read_cub_file(cub_path+'/bounding_boxes.txt')])
	co_num = len(co_order)
	test_num = len(test_order)
	train_num = len(train_order)

	### read images
	train_im = list()
	test_im = list()
	co_im = list()
	train_labs = list()
	test_labs = list()
	co_labs = list()
	train_bb = list()
	test_bb = list()
	total_num = co_num + test_num + train_num
	print('>>> Reading CUB from: ' + cub_path)
	widgets = ["CUB", Percentage(), Bar(), ETA()]
	pbar = ProgressBar(maxval=total_num, widgets=widgets)
	pbar.start()
	counter = 0

	### read content images
	for i in co_order:
		counter += 1
		pbar.update(counter)
		bbox = im_bbox[i]
		im = read_image(cub_path+'/images/'+im_fnames[i], co_size, 
			sqcrop=False, bbox=(bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))
		co_im.append(im)
		co_labs.append(im_class[i])

	### read test images
	for i in test_order:
		counter += 1
		pbar.update(counter)
		im, w, h = read_image(cub_path+'/images/'+im_fnames[i], im_size, 
			sqcrop=False, verbose=True)
		test_im.append(im)
		test_labs.append(im_class[i])
		bbox = im_bbox[i]
		### transform bbox
		im_scale_w = 1.0 * im_size / w
		im_scale_h = 1.0 * im_size / h
		bbox[0] = int(bbox[0] * im_scale_w)
		bbox[1] = int(bbox[1] * im_scale_h)
		bbox[2] = int(bbox[2] * im_scale_w)
		bbox[3] = int(bbox[3] * im_scale_h)
		test_bb.append(np.array(
			[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]).reshape(1,4))

	### read train images
	for i in train_order:
		counter += 1
		pbar.update(counter)
		im, w, h = read_image(cub_path+'/images/'+im_fnames[i], im_size, 
			sqcrop=False, verbose=True)
		train_im.append(im)
		train_labs.append(im_class[i])
		bbox = im_bbox[i]
		### transform bbox
		im_scale_w = 1.0 * im_size / w
		im_scale_h = 1.0 * im_size / h
		bbox[0] = int(bbox[0] * im_scale_w)
		bbox[1] = int(bbox[1] * im_scale_h)
		bbox[2] = int(bbox[2] * im_scale_w)
		bbox[3] = int(bbox[3] * im_scale_h)
		train_bb.append(np.array(
			[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]).reshape(1,4))

	return (np.array(co_im), co_labs), \
		(np.array(test_im), test_bb, test_labs), \
		(np.array(train_im), train_bb, train_labs)