# Human Presence Detection with Ultrasonic Proximity Sensor and Machine Learning
This is an Individual Project(Time Series Signal Analysis with Machine Learning) supervised by Prof. Dr. Andreas Pech
# Key Words
Raw Data retrieval from FIUS Red Pitaya Ultrasonic Sensor System, Time Series Data Analysis, Data Preprocessing, Data Modelling, Convolutional Neural Network Deployment, Training in MobileNetv2, Human Presence Detection, Redpitaya embedded System

# Data Acquisition
As this project's goal is to find the human presence in office environment using SONAR system (time series signal), Red Pitaya is used for embedded system for Ultrasonic sensor data acquisition purpose. RedPitaya Ultrasonic system is a powerful hardware and software platform designed for ultrasonic measurements. At the core of the RedPitaya Ultrasonic system is the RedPitaya board, which is a credit card-sized device that houses the digitizer and field-programmable gate array (FPGA) which has two high-speed analog input channels with sampling rates up to 125 MS/s and a resolution of 14 bits.

![image](https://github.com/ShafaitAzam/Deep-Learning-Project-1/assets/59325753/f236aa41-4d69-4b4d-9a7a-1193ad0dfe7b)

Fig: Red Pitaya Ultrasonic System used in the project.
In the Image below Office environment set up is shown. On the stand one Red Pitaya ultrasonic sensors were placed focusing on desk and chair. Sitting Standing person and different hard and soft surfaced chairs were used to take Person and object data. Then acquired 30k ADC signal data were used for model training testing and validation purpose.

![lab setup](https://github.com/javedcoding/Person-Detection-Using-SONOR-Signal-and-MobileNetV2/assets/59325753/b9402131-3116-4467-9945-25e2c05045e2)

Fig: Set up for Lab data acquisition

# Data Repository
As the data file is big the ADC data retrieved from SONAR sensor attached with Radpitaya can be downloaded from below links
Sensor: https://drive.google.com/file/d/1zcM6RURsS3SvCEv5lUTWFKrbN8u_Uw2H/view?ts=643eb1ca

# How to Train Machines
There are two machines for sensor to detect if there is a person present in the desk or not. So it is problem of binary classification for learning in CNN with time series signal. The main logic behind this project was to train two machines for the sensor to detect if the Own built CNN is better than MobileNetv2 in testing dataset.
First download two csv files for data acquired from the ADC signal of SONAR supported Radpitaya system. Then use the portion of data preprocessing to generate spectrograms. Here data file location have to be given as argument while calling "DataPreProcessing" class and by calling "save_Spectogram" method of that class we can save the spectrograms and numpy arrays of those spectograms into two seperated folders. After this step spectograms may look like this:

![image](https://github.com/javedcoding/Person-Detection-Using-SONOR-Signal-and-MobileNetV2/assets/59325753/558c2e94-d9f5-49b0-8a39-c9abf20e8957)


Here spectograms will help Human Eye to understand scenarios of the whole spectograms and numpy arrays of these spectograms are easier for computer to read for the next process. We have taken only the first 512 values of the frequency spectogram to take only significant changes into account of a whole time series signal for reducing computational hazards.


After doing the data preprocessing only by calling the class CNNMachine one can easily make the CNN machines with predefined parameters for 50k data for a project without changing any internal parameters. While initiating this class one must give the true csv file and the spectogram folder's path. after training by only calling the "predict" method one can easily get the result of prediction from a new data set if there is a person present before the sensor or not.

| Layer(Type)       | Output Shape |                    | Layer(Type)       | Output Shape |
|------------------------|----------------------|       |------------------------|----------------------|        
| conv2d_1 (Conv2d)      | (None, 254, 254, 32) |       | Block17 BatchNormal    | (None, 7, 7, 960)    |
| max_pooling2d_1        | (None, 127, 127, 32) |       | Block17 ReLU           | (None, 7, 7, 960)    |       
| conv2d_2               | (None, 125, 125, 64) |       | Block17 Depthw         | (None, 7, 7, 960)    |
| max_pooling2d_2        | (None, 62, 62, 64)   |       | Block17 BatchNormal    | (None, 7, 7, 960)    |
| dropout_1              | (None, 62, 62, 64)   |       | Block17 ReLU           | (None, 7, 7, 960)    |
| conv2d_3               | (None, 60, 60, 128)  |       | Block17 Conv2D         | (None, 7, 7, 320)    |
| max_pooling2d_3        | (None, 30, 30, 128)  |       | Block17 BatchNormal    | (None, 7, 7, 320)    |
| conv2d_4               | (None, 28, 28, 256)  |       | Block17 Conv2D         | (None, 7, 7, 1280)   |
| max_pooling2d_4        | (None, 14, 14, 256)  |       | Block17 BatchNormal    | (None, 7, 7, 1280)   |
| flatten_1              | (None, 50176)        |       | Block17 ReLU           | (None, 7, 7, 1280)   |
| dense_1                | (None, 64)           |       | Block17 GlobalAverage  | (None, 1280)         |
| dropout_2              | (None, 64)           |       | Block17 Dense          | (None, 1000)         |
| dense_2                | (None, 1)            |       

Tab: Convolutional Neural Network Composition           Tab: MobileNetV2 model Composition

# Result & Accuracy
As we have splitted the data set into 75% traning data set and 25% testing dataset, we got a lowest accuracy of 96.64% after several runs in different processors and kaggle. 

![Upper sensor confusion matrix](https://user-images.githubusercontent.com/59325753/234520566-049dc48b-f000-477f-a9dc-85d2eb3a32fe.jpg)

Fig: Accuaracy on Test Data in MobileNetV2 

As we have splitted the data set into 85% traning data set and 15% testing dataset, we got a lowest accuracy of 99.8% after several runs in different processors and kaggle. 
![image](https://github.com/javedcoding/Person-Detection-Using-SONOR-Signal-and-MobileNetV2/assets/59325753/42a5b2db-0670-40a3-ac7f-6e5d7e98e824)

Fig: Accuaracy on Test Data in own built CNN 

So we decided to proceed with own built CNN. In the seperate testing set of 2k data the accuracy was bellow 86%:

![image](https://github.com/javedcoding/Person-Detection-Using-SONOR-Signal-and-MobileNetV2/assets/59325753/0e63230a-3aec-46d5-8003-11218ffabbc4)


Fig: Confusion Maxtrix of Test Data in CNN

# Usage of The Project 
The purpose of this project is to get the weight for a CNN machine to prepare for an array of SONAR sensors where detection of sensor can tell if there is a person present or not in certain office environment along with annotated image data. For this reason detection of presence of a person is required which has been done in this project with satisfactory accuracy level. With the detection of the person Redpitaya internal C code is manipulated to turn on Lights. For the university owned code reason, C code inside Redpitaya is not possible to publish.     
