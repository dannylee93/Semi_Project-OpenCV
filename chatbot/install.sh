#!/bin/bash
cd .data
rm -r darknet
git clone https://github.com/pjreddie/darknet.git
cd darknet
make
wget https://pjreddie.com/media/files/darknet19.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights