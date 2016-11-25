import sys
import cv2
import argparse
from PIL import Image
import caffe
import matplotlib as plt

IMAGE_WIDTH = 256
IMAGE_HEIGHT =256
def transform_img(img,img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
	img[:,:,0] = cv2.equalizeHist(img[:,:,0])
	img[:,:,0] = cv2.equalizeHist(img[:,:,0])
	img[:,:,0] = cv2.equalizeHist(img[:,:,0])
	img = cv2.resize(img,(img_width,img_height),interpolation = 		cv2.INTER_CUBIC)	
	return img



'''
if __name__ == '__main__':

	parser = argparse.ArgumentParser(description = 'Apply heatmap model ')
	parser.add_argument()
'''

caffe.set_mode_cpu()

net = caffe.Net('matlab.prototxt','/home/cdeng//caffe-heatmap/data/flic/caffe-heatmap-flic.caffemodel',caffe.TEST) #Check python caffe API

im = cv2.imread('test.jpg')

transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
im = transform_img(im)
# im_input = im[np.newaxis, np.newaxis, :, :]



transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))

net.blobs['data'].data[...] = transformer.preprocess('data',im)
out = net.forward()
out['conv5_fusion'
