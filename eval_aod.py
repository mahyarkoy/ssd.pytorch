import pickle as pk
import numpy as np
import os

from data import AODDetection, BaseTransform
from condet_util import prep_voc_aod_data, prep_cub_aod_data
from calculate_mean_ap import get_avg_precision_at_iou

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0, 1" for multiple

'''
read the annotations and return list of shape [class_num+1, image_num, 4]
'''
def read_gt(dataset):
	num_images = len(dataset)
	gt_bbox = list()
	for i in range(len(dataset)):
		im_name, gt = dataset.pull_anno(i)
		gt_bbox.append([bc[:4] for bc in gt])

	return gt_bbox, num_images

'''
read detections.pk file saved by eval.py, construct data holders, and compute ap.
'''
def compute_average_precision(dets, dataset, save_path):
	### read gt bboxes
	print('>>> reading annotations')
	gt_bbox, im_num = read_gt(dataset)
	im_ids = range(im_num)

	### prepare bboxes
	r_bboxes_dict = dict(zip(im_ids, gt_bbox))
	g_bboxes_dict = dict(zip(im_ids, 
		[{"boxes": d[:, :4].tolist(), "scores": d[:, 4].tolist()} for d in dets]))
	data = get_avg_precision_at_iou(r_bboxes_dict, g_bboxes_dict, iou_thr=0.5)
	ap = data['avg_prec']
	print('>>> average precision: {:.4f}'.format(data['avg_prec']))

	### save results
	with open(save_path, 'w+') as fs:
		print('>>> average precision: {:.4f}'.format(ap) , file=fs)

	return ap

'''
Forward the net on dataset to collect the detections.
'''
def test_net(save_path, net, cuda, dataset):
	num_images = len(dataset)
	# all detections are collected into:
	#    all_boxes[cls][image] = N x 5 array of detections in
	#    (x1, y1, x2, y2, score)
	all_boxes = list()

	# timers
	_t = {'im_detect': Timer(), 'misc': Timer()}

	for i in range(num_images):
		### read data from dataset
		im, gt, h, w = dataset.pull_item(i)

		x = Variable(im.unsqueeze(0))
		if cuda:
			x = x.cuda()
		_t['im_detect'].tic()

		### run network on data, detections shape: (num_im, num_classes, num_boxes, 5)
		### the 5 nums per box are: (conf_score, xmin, ymin, xmax, ymax)
		detections = net(x).data
		detect_time = _t['im_detect'].toc(average=False)

		# skip j = 0, because it's the background class
		dets = detections[0, 1, :]
		mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
		dets = torch.masked_select(dets, mask).view(-1, 5)
		if dets.dim() == 0:
			continue
		boxes = dets[:, 1:]
		boxes[:, 0] *= w
		boxes[:, 2] *= w
		boxes[:, 1] *= h
		boxes[:, 3] *= h
		scores = dets[:, 0].cpu().numpy()
		cls_dets = np.hstack((boxes.cpu().numpy(),
							  scores[:, np.newaxis])).astype(np.float32,
															 copy=False)
		all_boxes.append(cls_dets)

		print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
													num_images, detect_time))

	with open(save_path, 'wb') as f:
		pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

	return all_boxes

if __name__ == '__main__':
	pred_save_path = '/media/evl/Public/Mahyar/ssd_logs/ssd_0/ssd300_120000/test/detections.pkl'
	save_path = '/media/evl/Public/Mahyar/ssd_logs/ssd_0/logs_ap.txt'
	model_path = '/media/evl/Public/Mahyar/ssd_logs/ssd_0/ssd300_120000'
	num_classes = 2
	cuda = True

	net = build_ssd('test', 300, num_classes)
	net.load_state_dict(torch.load(model_path))
	net.eval()
	print('Finished loading model!')

	voc_aod_path = '/media/evl/Public/Mahyar/condet_logs/47_logs_10stn_vocbird50/run_2/voc_bird_10_500_500_2.cpk'
	im_names, bboxes = prep_voc_aod_data(voc_aod_path, 'test')

	#cub_aod_path = '/media/evl/Public/Mahyar/condet_logs/46_logs_10stn_cub1shot/run_0/cub_split_1_5_50_0.cpk'
	#im_names, bboxes = prep_cub_aod_data(cub_aod_path)
	
	dataset_mean = (104, 117, 123)
	dataset = AODDetection(im_names, bboxes,
		transform=BaseTransform(300, dataset_mean))
	if cuda:
		net = net.cuda()
		cudnn.benchmark = True
	
	### detect
	all_boxes = test_net(pred_save_path, net, cuda, dataset)
	### evaluate
	compute_average_precision(all_boxes, dataset, save_path)
