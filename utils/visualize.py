import cv2
import matplotlib.pyplot as plt
import numpy as np

def visualize_from_log(log_info_path):
	loss = []
	pos_accuracy = []
	neg_accuracy = []
	with open(log_info_path, 'r') as f:
		while True:
			line = f.readline()
			if not line:
				break
			content = line.strip('\n').split(' ')
			print content
			neg_accuracy.append(float(content[-1].strip(',').split(':')[-1]))
			print content[-2].split(':')[-1]
			pos_accuracy.append(float(content[-2].strip(',').split(':')[-1]))
			loss.append(float(content[-3].strip(',').split(':')[-1]))

	index = range(len(loss))
	plt.figure(12)
	plt.subplot(221)
	plt.title('pos_accuracy')
	plt.plot(index, pos_accuracy)
	plt.subplot(222)
	plt.title('neg_accuracy')
	plt.plot(index, neg_accuracy)

	plt.subplot(212)
	plt.title('loss')
	plt.plot(index, loss)

	plt.show()


