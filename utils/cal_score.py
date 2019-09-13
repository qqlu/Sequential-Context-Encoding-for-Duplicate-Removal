import cv2
import numpy as np

def cal_pr_re_f1(predict, label):
	predict_true = np.where(predict == 1)[0]
	predict_false = np.where(predict == 0)[0]
	label_true = np.where(label == 1)[0]
	label_false = np.where(label == 0)[0]

	TP = np.intersect1d(predict_true, label_true)
	FP = np.intersect1d(predict_true, label_false)
	FN = np.intersect1d(predict_false, label_true)
	precision = len(TP) / (len(TP) + len(FP))
	recall = len(TP) / (len(TP) + len(FN))
	f1 = (2 * precision * recall) / (precision + recall)
	return [precision, recall, f1]


	


