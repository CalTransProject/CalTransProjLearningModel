# CalTrans Project Learning Model Environment  
Description:  
This is the demo version of the CalTrans Project real-time vehicle detection part built by the Computer Science Senior Design Project Team (led by student leader Ruitao Wu ) at California State University, Northridge on fall 2020 - spring 2021.
This version can process the video stream from the camera and detect the vehicles  

Environment is based on the:  
1. [opencv-python](https://opencv-python-tutroals.readthedocs.io/en/latest/)  
2. [numpy](https://www.numpy.org/), and [Flask](https://flask.palletsprojects.com/en/1.1.x/)  

Dowload the following files:  
1. [yolov3 weights](https://pjreddie.com/media/files/yolov3.weights)  
2. [yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true)  
3. [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)  

Place the yolov3.weights, yolov3.cfg, and coco.names in the same folder as the python script  
Install run the following command:  
`pip install flask`  
`pip install opencv-python`  
`pip install numpy`  
Clone the repository:  
`git clone https://github.com/RuitaoWu/CalTransProjLearningModel.git`  
`cd CalTransProjLearningModel`  
`python3 app.py`  
---  
Working Tree update  
```  
C:.
│   .gitignore
│   app.py
│   coco.names
│   config.ini
│   config.py
│   framemaker.py
│   GOPR0787.MP4
│   index.html
│   json_load.py
│   objRealTimeDectector.py
│   README.MD
│   result.json
│   test.mp4
│   viewcount.php
│   websocketdemo2.py
│   yolo.ipynb
│
├───data
│       test.csv
│
├───dumps
│       server.sql
│
├───static
│   │   app.js
│   │   chart.js
│   │   content.css
│   │   echarts.min.js
│   │   gallery.css
│   │   home.js
│   │   jQuery_mini.js
│   │   map.js
│   │   socket.io.min.js
│   │   socket.io.min.js.map
│   │   style.css
│   │   styles.css
│   │
│   └───images
│           .DS_Store
│           light gray.jpg
│           pic1.svg
│           point cloud.jpg
│           point cloud.svg
│           realtime-example.jpg
│
├───templates
│       index.html
│       websocketdemo2.html
│
├───YOLO
│       Readme
│       savayolo
│       yolov3.cfg
│       yolov3.weights
│
└───__pycache__
        config.cpython-39.pyc
        model.cpython-39.pyc
        objRealTimeDectector.cpython-39.pyc 
```  
