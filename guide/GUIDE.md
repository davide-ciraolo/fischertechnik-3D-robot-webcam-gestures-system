# Project Schema

The architecture of the project can be seen in the picture below:

![Project architecture.png](Project%20architecture.png?fileId=744018#mimetype=image%2Fpng&hasPreview=true)

Starting from the **robotic arm**, this is connected to the **PLC** that is the device that handles all the movements performed by the arm. It is done by a **ladder diagram** program written with the software TIA Portal, specific for programming these devices. The PLC is also connected with an ethernet wire to a **laboratory computer**, in wich the IP address is needed to be set with the one specified by TIA Portal in order to communicate with the PLC. The laboratory pc needs also to be connected to another wired or wireless network that can be used by the user computer in order to let them communicate between each other. In particular their addresses need to stay in the same subnet. The user computer (or laptop) need also to have a webcam connected or embedded to it. In this architecture the laboratory computer is directly connected to the PLC to control the robotic arm while the user computer takes care of detecting the user gestures through the webcam, convert them in commands and send them to the laboratory computer to perform the relative arm movements.

# Setup

In order to start the project, it is important to already have installed these software on the **laboratory computer**:

* Matlab (version 2020 or greater);
* TIA Portal (version 15 or greater);

And these software on the **user computer** (the laptop in the picture):

* Python 3 (version 3.7 or greater) with all the required libraries, in particular:
  * mediapipe (version 0.8.11 or greater);
  * opencv-python (version 4.6.0.66 or greater);
  * keras (version 2.10.0 or greater);
  * tensorflow (version 2.10.0 or greater);

In particular, in the final version of the gesture detection system, the keras and tensorflow libraries are not used anymore, because the detection part is handled by mediapipe and opencv libraries. The others two can also be used for experimental purpose in conjunction with the Cnn Gesture System class, but with the used dataset, the results of all the tested models are not good enough to be usable. The code related to this part is levaed in the project because it can be used with better models derived from a more meaningful dataset for this particular application.

# Laboratory pc and PLC connection

After ensured that the laboratory computer is connected with the PLC and the ethernet interface address of the computer is set up to **192.168.0.241** (the address specified in the TCON component of the TIA Portal ladder diagram), it is possible to start the connection by executing the "Matlab_PLC_connection.m" Matlab script. In this way Matlab will wait for the PLC in order to create the connection object. At this point it is enough to push the "Connection Start" button on the PLC dashboard to connect the two devices and create the connection object **t**. Once this object is created, the Matlab server and the PLC can communicate.

# Laboratory pc and user pc connection

The first step to estabilish a connection between the laboratory computer and the user computer is to ensure that both computers are connected to the same LAN. In this way they will have IP addresses of the same subnet and thus they can communicate. The connection between the laboratory computer and the user computer is estabilished with another Matlab server socket and a python client, respectively. The client is also the part of the project that is able to detect the hand gestures of the user through the client computer webcam. In particular the Matlab server can be opened executing the script named "server.m". It is important also to have the Matlab "Current Folder" view positioned on the "server.m" parent folder in order to be able to open all the required files, in particular the "connectionFcn.m" file. Once the "server.m" is executed, Matlab will wait for the client to connect to the socket. At this point it is enough that the user runs from the client computer the "app.py" script. After that, on Matlab the **srv** variable will be instantiated, representing the socket object, and on the command line will appear the message sent from the python client.

# Gesture detection

When all the devices are connected, in the user computer will appear a window containing the frames captured from the webcam. By moving the hand inside the camera frame, it is possible to see that the hand landmarks will be highlighted and at this point it is possible to make the gestures.

![gestures_vertical.png](gestures_vertical.png?fileId=744250#mimetype=image%2Fpng&hasPreview=true)