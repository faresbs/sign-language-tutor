from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.utils.misc import Timer

import torch 
import torch.nn as nn
from torch.autograd import Variable
import cnn_architectures as cnn
import numpy as np
import cv2
import argparse 

import sys
import time
import timeit
import joblib



def prep_image(image, isGray, input_dim, CUDA):

	img = cv2.resize(image, (input_dim, input_dim)) 
	

	if(isGray=='True'):
		img = np.resize(img, (input_dim, input_dim, 3))

	#Normalize test example
	if (isGray=='True'):

		mean = np.array([0.5, 0.5, 0.5])
		std = np.array([0.5, 0.5, 0.5])

	else:
		mean = np.array([0.485, 0.456, 0.406])
		std = np.array([0.229, 0.224, 0.225])


	img_ = std * img + mean

	img_ =  img_.transpose((2, 0, 1))
	img_ = img_[np.newaxis,:,:,:]/255.0

	#values outside the interval are clipped to the interval edges
	img_ = np.clip(img_, 0, 1)

	img_ = torch.from_numpy(img_).float()
	img_ = Variable(img_)

	if CUDA:
		img_ = img_.cuda()

	#Take only one channel for Gray since they are all duplicated channels
	if(isGray=='True'):
		img_ = img_[:, 0, :, :].unsqueeze(1)

	return img_


if __name__ == '__main__':


	parser = argparse.ArgumentParser(
		description='Run the class/detection network')

	parser.add_argument("--detection_path", default='weights/detection/mobilenet-v1-ssd-Epoch-200-Loss-3.0682483695802234.pth', type=str,
                    help="path of the detection model")

	parser.add_argument("--label_detection_path", default='weights/detection/hand.txt', type=str,
                    help="class labels of detection")

	parser.add_argument("--class_path", default='weights/class', type=str,
                    help="path of the classification model")

	parser.add_argument("--arch", default="vgg16", type=str,
                    help="classification architecture used")

	parser.add_argument("--image_size", default=224, type=int,
                    help="Size of the image")

	parser.add_argument("--threshold", default=0.5, type=float,
                    help="threshold of choosing a label class")

	parser.add_argument("--isGray", default=False, type=bool,
                    help="input images are RGB or grayscale")

	args = parser.parse_args()

	class_names_detection = [name.strip() for name in open(args.label_detection_path).readlines()]


	#Extract class names for the classification task
	with open(args.class_path+'/class_names', "rb") as file:
		class_names = joblib.load(file)

	num_classes = len(class_names)

	#if Gray images then number of channels is 1, if images are rgb then 3
	if (args.isGray==True):
		c = 1
	else:
		c = 3

	print('Handshape labels: '+str(class_names))

	# Device configuration
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	if device=='cpu':
		print ("Running on CPU.")
	elif device=='cuda:0':
		print ("Running on GPU.")



	#Loading classification model
	print("Loading networks...")
	if (args.arch == 'vgg16'):
		class_model = cnn.vgg16(num_classes=num_classes)
	elif (arch == 'inceptionv3'):
		class_model = cnn.Inception3(num_classes=num_classes, channels=c, aux_logits=True)

	class_model.load_state_dict(torch.load(args.class_path+'/weights.h5'))
	print("Classification Network successfully loaded.")
	    
	#put the model in eval mode to disable dropout
	class_model = class_model.eval()

	#load model into the gpu
	class_model = class_model.to(device)


	#Loading detection model
	detect_model = create_mobilenetv1_ssd(2, is_test=True)
	detect_model.load(args.detection_path)
	predictor = create_mobilenetv1_ssd_predictor(detect_model, candidate_size=200, device=device)
	print("Detection Network successfully loaded.")

	video_capture = cv2.VideoCapture(0)


	#Empty label
	empty='None'

	#add some space for the detected bounding box
	add_bbox = 30

	#Take only two hands and ignore the rest
	#UNTIL I FIND A THEORETICAL SOLUTION
	#Take only one hand : greedy approach, the hand in the most right position
	limit = False
	size = 1

	while True:

		# Capture frame-by-frame
		ret, frame = video_capture.read()

		if(ret == False):
			print('Web cam is not found!')
			sys.exit()

		#Inverse frame
		frame = cv2.flip(frame, 1)

		#image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		#Start timer
		start_total = timeit.default_timer()

		#Predict bounding boxes using the first sub network
		boxes, labels, probs = predictor.predict(frame, 10, 0.4)

		#Extract images from bounding box
		images = []
		x1s = []
		y1s = []


		#TO FIX : bbox keep jumping from hand to hand (is there an easy solution for this?)
		if (boxes.size(0) > size)&(limit):
			boxes = boxes[0:size, :]


		for i in range(boxes.size(0)):

			#Start timer
			start = timeit.default_timer()

			#take box by box
			box = boxes[i, :]

			#Transform from tensor to int and extend the bbox
			x1 = int(box[0]) - add_bbox
			y1 = int(box[1]) - add_bbox
			x2 = int(box[2]) + add_bbox
			y2 = int(box[3]) + add_bbox

			

			#coords must not exceed the limit of the frame or be negative
			if x1 < 0:
				x1 = 0

			if x2 > frame.shape[1]:
				x2 = frame.shape[1]

			if y1 < 0:
				y1 = 0

			if y2 > frame.shape[0]:
				y2 = frame.shape[0]


			#extract th wanted image from the full frame
			image = frame[y1:y2, x1:x2]

			if (args.isGray==True):
				image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			#cv2.imshow("image", image)

			#Resize captured image to be identical with the image size of the training data
			img = prep_image(image, args.isGray, args.image_size, CUDA=True)

			#Prediction class label
			output = class_model(img)
			prediction = output.data.argmax()
			value, predicted = torch.max(output.data, 1)


			#Tranform logits to probablities
			m = nn.Softmax()
			input = output.data
			output = m(input)
			output = np.around(output,2)
			value, predicted = torch.max(output.data, 1)


			#if prediction is not accurate returns empty
			if (value >= args.threshold):
				prediction = class_names[predicted]
				fontColor = (255, 0, 0)
				lineType = 4
			else:
				prediction = empty
				fontColor = (255, 255, 0)
				lineType = 2


			#Display bounding box with text
			cv2.rectangle(frame, (x1, y1), (x2, y2), fontColor, lineType)
			cv2.putText(frame, prediction, (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)  

			stop = timeit.default_timer()
			inference = (stop - start)
			print("Inference time total: {:.4f}".format(inference))


		stop_total = timeit.default_timer()
		running_time = 1/(stop_total - start_total)

		print ("FPS: {:.2f}, Found {:d} objects".format(running_time, len(probs)))
		running_time = int(running_time)
		cv2.putText(frame, str(running_time) + " FPS", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)  
					
		
		cv2.imshow("Video Stream", frame)


		#PRESS Q TO QUIT
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	video_capture.release()
	cv2.destroyAllWindows()

		