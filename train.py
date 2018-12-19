import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import utils, datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import joblib
import datetime as dt
import time
import os
import copy
from time import sleep

np.random.seed(0)

#Display learning curve 
def learning_curve(train_acc, val_acc, train_loss, val_loss, path):

	epochs = range(len(train_acc))

	plt.figure()

	plt.plot(epochs, train_acc, 'b', label='Training acc')
	plt.plot(epochs, val_acc, 'g', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.xlabel('epoch')
	plt.legend()
	plt.savefig(path+"accuracy.png")

	plt.figure()

	plt.plot(epochs, train_loss, 'b', label='Training loss')
	plt.plot(epochs, val_loss, 'g', label='Validation loss')
	plt.title('Training and validation loss')
	plt.xlabel('epoch')
	plt.legend()
	plt.savefig(path+"loss.png")


def imshow(inp, title=None):
	    inp = inp.numpy().transpose((1, 2, 0))
	    mean = np.array([0.485, 0.456, 0.406])
	    std = np.array([0.229, 0.224, 0.225])
	    inp = std * inp + mean
	    inp = np.clip(inp, 0, 1)
	    plt.imshow(inp)
	    if title is not None:
	        plt.title(title)
	    plt.pause(10)  # pause a bit so that plots are updated
	    plt.close() 


def train(data_dir, tune=True):

	#When the model started training
	start_date = dt.datetime.now().strftime("%Y-%m-%d-%H:%M")
	print ("Start Time: "+start_date)

	#Load model architecture
	if(tune):
		print ("pretrained model loaded..")
		model = models.vgg16(pretrained='imagenet')
	else:
		model = models.vgg16()


	##Replace the last out layer from 1000 classes to num classes

	# Number of filters in the bottleneck layer
	#num_ftrs = model.classifier[-1].in_features
	# convert all the layers to list and remove the last one
	#features = list(model.classifier.children())[:-1]
	## Add the last layer based on the num of classes in our dataset
	#features.extend([nn.Linear(num_ftrs, 24)])
	## convert it into container and add it to our model class.
	#model.classifier = nn.Sequential(*features)


	##Use model.classifier[0].in_features instead for vgg16
	##Use model.fc.in_features for other models
	##Replace the last FC layers with initial weights
	num_in_ftrs = model.classifier[0].in_features
	num_out_ftrs = model.classifier[0].out_features
	model.classifier = nn.Sequential(
            nn.Linear(num_in_ftrs, num_out_ftrs),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(num_out_ftrs, num_out_ftrs),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(num_out_ftrs, 24))

	if(tune):
		#Freeze some layers
		ct = 0
		for child in model.children():
			for param in model.parameters():
				if(ct < 15):
					param.requires_grad = False
				ct += 1

		# To view which layers are freezed and which layers are not
		#layer = 0
		#print("Trainable layers:\n")
		#for params in model.parameters():
		#	print("Layer "+str(layer)+": "+str(params.requires_grad))
		#	layer += 1

	#else:
		# freeze all layers
	#	for param in model.parameters():
	#		param.requires_grad = False

	print (model)

	print("\nLoading images with data augmentation on the fly..")


	# Data augmentation and normalization for training
	# Just normalization for validation
	data_transforms = {
	    'train': transforms.Compose([
	        #transforms.RandomResizedCrop(224),
	        #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.15, hue=0.1),
	        #transforms.ColorJitter(brightness=0.2, contrast=0.2),
	        transforms.RandomRotation(10, resample=False, expand=False, center=None),
	        transforms.RandomHorizontalFlip(),
	        transforms.ToTensor(),
	        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	        
	    ]),
	    'val': transforms.Compose([
	        #transforms.Resize(256),
	        #transforms.CenterCrop(224),
	        transforms.ToTensor(),
	        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	    ]),
	}

	
	image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
	        			for x in ['train', 'val']}
	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
	                                             	shuffle=True, num_workers=4)
	              for x in ['train', 'val']}

	#Augment the size of data 
	multiplier = 2
	dataset_sizes = {}
	
	dataset_sizes['train'] = len(image_datasets['train']) * multiplier
	dataset_sizes['val'] = len(image_datasets['val'])

	class_names = image_datasets['train'].classes
	print (dataset_sizes)


	#Visualize random images
	for i in range(1):
		# Get a batch of training data
		inputs, classes = next(iter(dataloaders['train']))

		# Make a grid from batch
		out = utils.make_grid(inputs)

		imshow(out, title="Training images")


	#Visualize random images
	for i in range(1):
		# Get a batch of val data
		inputs, classes = next(iter(dataloaders['val']))

		# Make a grid from batch
		out = utils.make_grid(inputs)

		imshow(out, title="Validation images")


	# Device configuration
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	model = model.to(device)

	num_epochs = 50

	criterion = nn.CrossEntropyLoss()

	# Observe that all parameters are being optimized
	if(tune):
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, momentum=0.9, weight_decay=0.0005)
	else:
		optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

	# Decay LR by a factor of 0.1 every 7 epochs
	#scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	#to save results
	train_acc = []
	train_loss = []
	val_acc = []
	val_loss = []

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 30)

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:

			if phase == 'train':
				
				#scheduler.step()
				model.train()  # Set model to training mode

				print ("Training..")

			else:
				model.eval()   # Set model to evaluate mode
				print ("Evaluating..")

			running_loss = 0.0
			running_corrects = 0

			#Progress bar to visualize training progress
			import progressbar
			
			bar = progressbar.ProgressBar(maxval=dataset_sizes[phase], widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
			bar.start()
			i = 0		    

			# Iterate over data.
			for inputs, labels in dataloaders[phase]:
				inputs = inputs.to(device)
				labels = labels.to(device)

				#For progress bar
				i += len(inputs)
				bar.update(i)
				sleep(0.01)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs)
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)

					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

				# statistics
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)

			bar.finish()

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]

			print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            		phase, epoch_loss, epoch_acc))

			#Save results

			if(phase=='train'):
				train_acc.append(epoch_acc)
				train_loss.append(epoch_loss)
			elif(phase=='val'):
				val_acc.append(epoch_acc)
				val_loss.append(epoch_loss)

        	# deep copy the model
        	if (phase == 'val') and (epoch_acc>best_acc):
        		best_acc = epoch_acc
        		best_model_wts = copy.deepcopy(model.state_dict())



    #date of the model stoped Training
	end_date = dt.datetime.now().strftime("%Y-%m-%d-%H:%M")

	print ("Start Time: "+start_date)
	print("End Time: "+end_date)

	#Make a directory to save the weights according to date
	try:
		os.makedirs('weights/'+end_date)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			raise

	#Saving learning curve
	learning_curve(train_acc, val_acc, train_loss, val_loss, 'weights/'+end_date+'/')

	#Saving class names
	with open('weights/'+end_date+"/class_names", "wb") as file:
		joblib.dump(class_names, file) 

	#Saving the model weights
	torch.save(model.state_dict(), 'weights/'+end_date+'/'+'weights.h5')

	return model

if __name__ == '__main__':
	#PATH = 'small_files/ASL'
	#train(PATH)