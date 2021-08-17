# facenet-poc

This repository contains a demonstration of face recognition using the FaceNet network (https://arxiv.org/pdf/1503.03832.pdf) and a webcam. Our implementation feeds frames from the webcam to the network to determine whether or not the frame contains an individual we recognize.

Install
-----

## clone the repository
    > https://github.com/mlpocprojects/facenet-poc.git
    
    # change the directory
    > cd facenet-poc

Create a virtualenv in the facenet-poc directory and activate it::
    
    > python -m venv venv
    > venv\Scripts\activate.bat
    
Install Dependencies in Virtual Environment::

	> pip install -r requirements.txt

RUN
---

 On Local Virtual Environment::

	> python webstreamer.py
	
Open http://localhost:8000 in a browser.

# important
-----------

    $ set OPENCV_OPENCL_DEVICE=disabled

This will prevent the memory full error on openGL (i.e: for liveness detection)