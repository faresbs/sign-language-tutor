# Sign Language Tutor

An intelligent application to teach sign language to non-deaf users. The application have a recongnition system that recognise and evaluate the user's gestures.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

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

Install denpendencies for the recognition system. We used Pytorch as a ML library, OpenCv2 for computer vision tasks, the application requires GPU to work properly and fast but can also work on CPU too.   

```
```


### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Run recognition system for hand detection + handshape classification
Recognition system is composed of two sub models: the first is for hand detection and the second hand classification. 

### Pretrained Models
hand detection model (Mobilenetv1-ssd)
URL: https://drive.google.com/file/d/1dhjs9WJmQIirgxC0u47DnUqynBAm3B7r/view?usp=sharing

hand classification model (VGG16)
URL: https://drive.google.com/file/d/12-SDr-KZ3I1tYXtsTYHScTxq2SS5Ig4j/view?usp=sharing

## Running

You can test out the recognition system directly wihout.

```
python3 run.py
```


## Built With

* [Kivy](https://kivy.org/#home) - Open source Python library for UI developement
* [Pytorch](https://pytorch.org/) - ML library
* [Opencv](https://opencv.org/) - Open Source Computer Vision Library

## Contributing

You are free to use this project or contribute that would be cool.

## Authors

* **Fares Ben Slimane** - *recognition system* - [check my personal webpage](http://faresbs.github.io)
* **Gaieth Dkhili** - *UI application*)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* This project was made 


