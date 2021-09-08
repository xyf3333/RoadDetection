
# Road Detection

## Project overview

The purpose is detecting the yellow strip on the road and fellowing yellow strip using camera realtime data. 

## Table of Contents

1. [Installation](#installation)
2. [QuickStart](#quickstart)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation 

The code should run using Python version 3.*.

The necessary libraries are: numpy, rospy, adafruit_servokit, std_msgs, OpenCV

## File Descriptions

README.md
talker.py
Reading road data from camera,  processing and publishing real time messages. 
listener.py
Revicing the messages and cantrol the motor using servoKit.
LICENSE

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
The Data set used in this process is provided by **figure-8** that contains real labeled disaster messages received by an aid organisation during disaster events.
