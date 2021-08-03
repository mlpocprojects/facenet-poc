# facenet-poc

This repository contains a demonstration of face recognition using the FaceNet network (https://arxiv.org/pdf/1503.03832.pdf) and a webcam. Our implementation feeds frames from the webcam to the network to determine whether or not the frame contains an individual we recognize.

## How to use

To install all the requirements for the project run

	pip install -r requirements.txt

In the root directory. After the modules have been installed you can run the project by using python

	python webstreamer.py
	
# important
-----------

    $ set OPENCV_OPENCL_DEVICE=disabled

This will prevent the memory full error on openGL (i.e: for liveness detection)