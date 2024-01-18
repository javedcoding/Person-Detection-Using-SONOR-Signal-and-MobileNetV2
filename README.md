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
As the data file is big the ADC data retrieved from two SONAR sensors attached with Radpitaya can be downloaded from below links
Lower sensor: https://drive.google.com/file/d/1zcM6RURsS3SvCEv5lUTWFKrbN8u_Uw2H/view?ts=643eb1ca
Upper sensor: https://drive.google.com/file/d/1Jxn6NIRaSjZDK_I2HiUXpHcbwpzgeb2a/view?ts=643eb1f2

# How to Train Machines
There are two machines for two sensors to detect if there is a person present in the seat or not. So it is problem of binary classification. The main logic behind this project was to train two machines for two of the sensors (upper and lower) to detect if there is a full sized human present covering both upper and lower sensors or there is a small infant covering only the lower sensor or there is no one in the seat (explained in details in the documentation section).
First download two csv files for data acquired from the ADC signal of SONAR supported Radpitaya system. Then use the portion of data preprocessing to generate spectrograms. Here data file location have to be given as argument while calling "DataPreProcessing" class and by calling "save_Spectogram" method of that class we can save the spectrograms and numpy arrays of those spectograms into two seperated folders. After this step spectograms may look like this:

![Lower sensor spectrogram example picture](https://user-images.githubusercontent.com/59325753/234520045-3151bfe1-d1aa-4644-9209-117b812f83c7.jpg)

Here spectograms will help Human Eye to understand scenarios of the whole spectograms and numpy arrays of these spectograms are easier for computer to read for the next process. We have taken only the first 512 values of the frequency spectogram to take only significant changes into account of a whole time series signal for reducing computational hazards.


After doing the data preprocessing only by calling the class CNNMachine one can easily make the CNN machines with predefined parameters for 50k data for a project without changing any internal parameters. While initiating this class one must give the true csv file and the spectogram folder's path. after training by only calling the "predict" method one can easily get the result of prediction from a new data set if there is a person present before the sensor or not.

| Layer(Type)       | Output Shape |
|------------------------|----------------------|
| conv2d_1 (Conv2d)      | (None, 254, 254, 32) |
| max_pooling2d_1        | (None, 127, 127, 32) |
| conv2d_2               | (None, 125, 125, 64) |
| max_pooling2d_2        | (None, 62, 62, 64)   |
| dropout_1              | (None, 62, 62, 64)   |
| conv2d_3               | (None, 60, 60, 128)  |
| max_pooling2d_3        | (None, 30, 30, 128)  |
| conv2d_4               | (None, 28, 28, 256)  |
| max_pooling2d_4        | (None, 14, 14, 256)  |
| flatten_1              | (None, 50176)        |
| dense_1                | (None, 64)           |
| dropout_2              | (None, 64)           |
| dense_2                | (None, 1)            |

Tab: Convolutional Neural Network Composition

# Result & Accuracy
As we have splitted the data set into 85% traning data set and 15% testing dataset, we got a lowest accuracy of 99.8% after several runs in different processors and kaggle. 
A confusion Matrix for the Upper Sensor is similar to this:

![Upper sensor confusion matrix](https://user-images.githubusercontent.com/59325753/234520566-049dc48b-f000-477f-a9dc-85d2eb3a32fe.jpg)

Fig: Confusion Maxtrix of Test Data in CNN Machine1 for Upper Sensor

And the confusion matrix for the Lower Sensor can be similar to this:

![Lower sensor confusion matrix](https://user-images.githubusercontent.com/59325753/234520470-f8ebe1f7-5677-4653-92e9-7bebeb34bc7e.jpg)

Fig: Confusion Maxtrix of Test Data in CNN Machine2 for Lower Sensor

We have tested only 500 combined data, captured while both sensors were active. Among them 250 data was for regular sized human, 200 was for infant or very small and 50 was for no person. In the table I, we can see that 248 out of 250 regular sized humans were detected correctly. So, accuracy in this case 99.2%. In case of infant/very small person, 195 out of 200 were detected correctly by lower sensor and upper sensor detected nothing (means label 0). So, the accuracy is 97.5%. And in the case of no person, 45 out of 50 have been detected correctly.

![image](https://github.com/ShafaitAzam/Deep-Learning-Project-1/assets/59325753/f49f7b57-c2fd-4ea5-b6eb-776d8353d4ee)

Tab: Height Estimation Result

# Usage of The Project 
The purpose of this project is to get the weight for a CNN machine to prepare for an array of SONAR sensors where each detection of each sensor can tell a certain height deference of the passenger or driver of the seat. For this reason detection of presence of a person is required which has been done in this project with satisfactory accuracy level. With the detection of the height that car seat can be automatically adjusted for proper view for the driver and the passenger.     
