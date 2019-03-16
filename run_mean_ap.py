import pickle as pk
import numpy as np

from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
from data import VOC_CLASSES as labelmap
from calculate_mean_ap import get_avg_precision_at_iou

'''
read the annotations and return list of shape [class_num+1, image_num, 4]
'''
def read_gt(voc_dir):
	set_type = 'test'
	dataset_mean = (104, 117, 123)
	dataset = VOCDetection(voc_dir, [('2007', set_type)],
						   BaseTransform(300, dataset_mean),
						   VOCAnnotationTransform())

	num_images = len(dataset)
	gt_bbox = [[[] for _ in range(num_images)]
					for _ in range(len(labelmap)+1)]
	for i in range(len(dataset)):
		im_name, gt = dataset.pull_anno(i)
		for box_conf in gt:
			gt_bbox[box_conf[4]+1][i].append(box_conf[:4])

	return gt_bbox, num_images

'''
read detections.pk file saved by eval.py, construct data holders, and compute ap.
'''
def compute_average_precision(det_path, voc_dir, save_path):
	### read detections
	with open(det_path, 'rb') as fs:
		dets = pk.load(fs)
	class_num = len(labelmap)+1

	### read gt bboxes
	print('>>> reading annotations')
	gt_bbox, im_num = read_gt(voc_dir)
	im_ids = range(im_num)

	### prepare bboxes
	ap_list = list()
	print('>>> class names: ' + str(labelmap))
	for c in range(1, class_num):
		print('>>> computing for class: {}/{}'.format(c, class_num))
		r_bboxes_dict = dict(zip(im_ids, gt_bbox[c]))
		g_bboxes_dict = dict(zip(im_ids, 
			[{"boxes": d[:, :4].tolist(), "scores": d[:, 4].tolist()} for d in dets[c]]))
		data = get_avg_precision_at_iou(r_bboxes_dict, g_bboxes_dict, iou_thr=0.5)
		ap_list.append(data['avg_prec'])
		print('>>> average precision: {:.4f}'.format(data['avg_prec']))

	print('>>> mean AP: {:.2}'.format(100*np.mean(ap_list)))

	### save results
	ap_str = str(['{:.4f}'.format(ap) for ap in ap_list])
	with open(save_path, 'w+') as fs:
		print('>>> mean AP: {:.2f}'.format(100*np.mean(ap_list)), file=fs)
		print('>>> average precision: ' + ap_str, file=fs)
		print('>>> class names:' + str(labelmap), file=fs)

	return


if __name__ == '__main__':
	voc_dir = '/media/evl/Public/Mahyar/Data/voc/VOCdevkit/'
	det_path = '/media/evl/Public/Mahyar/ssd_logs/ssd_0/ssd300_120000/test/detections.pkl'
	save_path = '/media/evl/Public/Mahyar/ssd_logs/ssd_0/logs_ap.txt'
	compute_average_precision(det_path, voc_dir, save_path)