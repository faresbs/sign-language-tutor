# Sign Language Tutor

An intelligent application to teach sign language to non-deaf users. The application have a recongnition system that recognise and evaluate the user's gestures.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

Ofcourse, you need to have python, here we are using python 3.6. So you need to installe python3.

```
sudo apt-get update
sudo apt-get install python3.6
```

Install kivy dependencies for UI environment. 

```
1. Add one the PPAs:
	$ sudo add-apt-repository ppa:kivy-team/kivy

2. Update your package list using your package manager
	$ sudo apt-get update

3. Install Kivy

*Python2 - python-kivy:
	$ sudo apt-get install python-kivy

*Python3 - python3-kivy:
	$ sudo apt-get install python3-kivy
```

Install Pytorch a very cool machine learning library. 

```
https://pytorch.org/
```

Install opencv2.
```
sudo apt-get install python-opencv
```

Other dependencies (joblib, numpy, etc..).
```
pip install joblib
```
```
pip install numpy
```

## Test

### Run the recognition system for Hand detection + gesture recognition
Recognition system is composed of two sub models: the first is for hand detection and the second one is for hand classification. 

### Pretrained Models
hand detection model (Mobilenetv1-ssd)
URL: https://drive.google.com/file/d/1dhjs9WJmQIirgxC0u47DnUqynBAm3B7r/view?usp=sharing

hand classification model (VGG16)
URL: https://drive.google.com/file/d/12-SDr-KZ3I1tYXtsTYHScTxq2SS5Ig4j/view?usp=sharing

### Running

You need to download the models' files first. Place the classification model file (VGG16) under the weights/class folder. Place the detection model file (mobilenet-ssd) under the weights/detection folder.

You can test out the recognition system directly wihout accessing to the application UI. It will run by default on GPU, but nevertheless, if you don't have the CUDA environment installed it will run on CPU. Keep in mind that it will run much smoother and faster on GPU.

```
python3 run.py
```
### Run the application

You can run the whole application with the graphical interface.

```
python3 main.py
```

## Screenshots

![Test Image 1](https://github.com/faresbs/sign-language-tutor/blob/master/screenshots/start.png)
![Test Image 2](https://github.com/faresbs/sign-language-tutor/blob/master/screenshots/app.png)

## Built With

* [Kivy](https://kivy.org/#home) - Open source Python library for UI developement
* [Pytorch](https://pytorch.org/) - ML library
* [Opencv](https://opencv.org/) - Open Source Computer Vision Library

## Contributing

You are free to use this project or contribute that would be cool.

## Authors

* **Fares Ben Slimane** - *recognition system* - [check my personal webpage](http://faresbs.github.io)
* **Ghaith Dekhili** - *UI application*

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* This project was made as part of the course of Intelligent Tutoring System.
* The detection ssd model has been taken from qfgaohao (https://github.com/qfgaohao/pytorch-ssd).  
* Background music used (https://opengameart.org/content/driving-0).


