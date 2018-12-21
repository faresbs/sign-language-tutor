##This is for creating new data images for sign language through sessions##
##Work for both temporal and static signs

from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor

import cv2
import datetime as dt
import time
import os
import numpy as np
import argparse

import torch 
import torch.nn as nn


if __name__ == "__main__":

	#add some space for the detected bounding box
	add_bbox = 10

	window_size = 400

	fontColor = (255, 255, 0)
	lineType = 2

	#hand detection model
	detection_path = 'weights/detection/mobilenet-v1-ssd-Epoch-200-Loss-3.0682483695802234.pth'
	

	#folder where we save data
	if not os.path.exists('data'):
	   os.makedirs('data')


	parser = argparse.ArgumentParser(
		description='Create new dataset for sign language')

	parser.add_argument("--sl", default="asl", type=str,
                    help="What's the sign language you'll be using?")

	parser.add_argument("--type", default="static", type=str,
                    help="temporal or static signs?")

	parser.add_argument("--sign", default="a", type=str,
                    help="What's the sign that you'll be using ? ")

	parser.add_argument("--bbox", default="small", type=str,
                    help="What's the bbox dimensions that you'll be using (big/small) ?")

	parser.add_argument("--size", default=299, type=int,
                    help="Size of the image")

	parser.add_argument("--auto", default=False, type=bool,
                    help="Automatic hand detection")

	args = parser.parse_args()


	#create a folder for the sign language
	if not os.path.exists('data/'+args.sl):
	   os.makedirs('data/'+args.sl)


	#Save the images in a folder session
	date = dt.datetime.now().strftime("%Y-%m-%d")
	if not os.path.exists('data/'+args.sl+'/'+date):
	   os.makedirs('data/'+args.sl+'/'+date)



	video_capture = cv2.VideoCapture(0)

	#For static signs
	if (args.type=='static'):

		if (args.auto == False):

			if (args.bbox == 'big'):
				#Dimension of the rectangle for the bounding box for multi-components sign
				x = 300
				y = 100
				h = 300
				w = 300
			elif (args.bbox == 'small'):
				#Dimension of the rectangle for the bounding box for handshape
				x = 100
				y = 150
				h = 200
				w = 200

		#Detect boundaries of hand using a detection model
		else:
			#Loading detection model
			detect_model = create_mobilenetv1_ssd(2, is_test=True)
			detect_model.load(detection_path)
			
			# Device configuration
			device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
			if device=='cpu':
				print ("Running on CPU.")
			elif device=='cuda:0':
				print ("Running on GPU.")

			predictor = create_mobilenetv1_ssd_predictor(detect_model, candidate_size=200, device=device)
			print("Detection Network successfully loaded...")

			

		#where we want to save the image
		save = 'data/'+args.sl+'/'+date+'/static/'+args.sign

		#Save in the sign folder in the current session folder
		date = dt.datetime.now().strftime("%Y-%m-%d")
		if not os.path.exists(save):
		   os.makedirs(save)

		if not os.path.exists(save):
		   os.makedirs(save)


		files = os.listdir(save)

		#get number of current images
		pic_num = len(files)

		#Initialize captured image as a black image
		image = np.zeros((window_size,window_size, 3), np.uint8)

		while True:

			#PRESS Q TO QUIT
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

			# Capture frame-by-frame
			ret, frame = video_capture.read()

			#Predict bounding boxes 
			if (args.auto):

				boxes, labels, probs = predictor.predict(frame, 10, 0.4)

				if(boxes.size(0) > 1):
					print("DO not support many detection!!")
					boxes = box = boxes[0, :]

				elif(boxes.size(0) == 1):
					boxes = boxes[0]

				else:
					continue

				#Transform from tensor to int and extend the bbox
				x1 = int(boxes[0]) - add_bbox
				y1 = int(boxes[1]) - add_bbox
				x2 = int(boxes[2]) + add_bbox
				y2 = int(boxes[3]) + add_bbox		

				#coords must not exceed the limit of the frame or be negative
				if x1 < 0:
					x1 = 0

				if x2 > frame.shape[1]:
					x2 = frame.shape[1]

				if y1 < 0:
					y1 = 0

				if y2 > frame.shape[0]:
					y2 = frame.shape[0]

				image = frame[y1:y2, x1:x2]

				#PRESS Space to capture the image
				k = cv2.waitKey(33)
				if k == 32:

					#resize before saving
					image = cv2.resize(image, (args.size, args.size)) 

					cv2.imwrite(save+'/'+str(pic_num)+'.png', image)
					pic_num += 1

				cv2.rectangle(frame, (x1, y1), (x2, y2), fontColor, lineType)

			else:

				#PRESS Space to capture the image
				k = cv2.waitKey(33)
				if k == 32:
					#Extract bounding box image
					image = frame[y:y+h,x:x+w]

					#resize before saving
					image = cv2.resize(image, (args.size, args.size)) 

					cv2.imwrite(save+'/'+str(pic_num)+'.png', image)
					pic_num += 1

				cv2.rectangle(frame, (x, y), (x+w, y+h), fontColor, lineType)


			#Display results
			frame = cv2.resize(frame, (window_size, window_size))
			image = cv2.resize(image, (window_size, window_size))
			cv2.imshow("Video Stream", np.hstack([frame, image]))

			


	#For temporal signs that require movement
	else:

		#where we want to save the image
		save = 'data/'+args.sl+'/'+date+'/temporal/'+args.sign

		#Save in the sign folder in the current session folder
		date = dt.datetime.now().strftime("%Y-%m-%d")

		if not os.path.exists(save):
		   os.makedirs(save)


		subfolders = os.listdir(save)
		folder_num = len(subfolders)

		#Initialize captured image as a black image
		image = np.zeros((window_size,window_size,3), np.uint8)

		while True:

			# Capture frame-by-frame
			ret, frame = video_capture.read()


			#Predict bounding boxes 
			if (args.auto):

				#Detect boundaries of hand using a detection model
			
				#Loading detection model
				detect_model = create_mobilenetv1_ssd(2, is_test=True)
				detect_model.load(detection_path)
				
				# Device configuration
				device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
				if device=='cpu':
					print ("Running on CPU.")
				elif device=='cuda:0':
					print ("Running on GPU.")

				predictor = create_mobilenetv1_ssd_predictor(detect_model, candidate_size=200, device=device)
				print("Detection Network successfully loaded...")

				boxes, labels, probs = predictor.predict(frame, 10, 0.4)

				if(boxes.size(0) > 1):
					print("DO not support many detection!!")
					boxes = box = boxes[0, :]

				elif(boxes.size(0) == 1):
					boxes = boxes[0]

				else:
					continue

				#Transform from tensor to int and extend the bbox
				x1 = int(boxes[0]) - add_bbox
				y1 = int(boxes[1]) - add_bbox
				x2 = int(boxes[2]) + add_bbox
				y2 = int(boxes[3]) + add_bbox		

				#coords must not exceed the limit of the frame or be negative
				if x1 < 0:
					x1 = 0

				if x2 > frame.shape[1]:
					x2 = frame.shape[1]

				if y1 < 0:
					y1 = 0

				if y2 > frame.shape[0]:
					y2 = frame.shape[0]

				frame = frame[y1:y2, x1:x2]

				#KEEP PRESSING k to save the frames in the folder 
				k = cv2.waitKey(1)
				if k == 32:
					if not os.path.exists(save+'/'+str(folder_num)):
						os.makedirs(save+'/'+str(folder_num))

					frame_num = 0
					while True:
						#Save in lower fps rate so that we dont capture unecessary frames
						k = cv2.waitKey(2000)
						if k == 32:
							cv2.imwrite(save+'/'+str(folder_num)+'/'+str(frame_num)+'.png', frame)
							frame_num += 1
							ret, frame = video_capture.read()
						else:
							folder_num += 1
							break
			
			cv2.rectangle(frame, (x1, y1), (x2, y2), fontColor, lineType)

			else:

				#KEEP PRESSING k to save the frames in the folder 
				k = cv2.waitKey(1)
				if k == 32:
					if not os.path.exists(save+'/'+str(folder_num)):
						os.makedirs(save+'/'+str(folder_num))

					frame_num = 0
					while True:
						#Save in lower fps rate so that we dont capture unecessary frames
						k = cv2.waitKey(2000)
						if k == 32:
							cv2.imwrite(save+'/'+str(folder_num)+'/'+str(frame_num)+'.png', frame)
							frame_num += 1
							ret, frame = video_capture.read()
						else:
							folder_num += 1
							break


			#Display results
			frame = cv2.resize(frame, (window_size, window_size))
			image = cv2.resize(image, (window_size, window_size))
			cv2.imshow("Video Stream", np.hstack([frame, image]))

			#KEEP PRESSING Q TO QUIT
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break










