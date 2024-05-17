[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/NTOOU2-8)
# A Resource-Constrained Surveillance System For Resource-Constrained Environments

### Introduction
Proliferation of Internet of Things (IoT) and Cyber-Physical Systems (CPS) has resulted in the developments and deployments of various applications for various purposes. The area of security and surveillance has received lots of attention because of the importance and the need to secure important or critical assets [1]. 

One seeming assumption about these systems is that they often assumed availability of reasonably enough resources. Such resources as computing capabilities, memory availability, and high bandwidth internet. However, developing and under-developing nations have resource constraints which is as results of factors associated with economic and technological capabilities of such settings [2]. Although disadvantaged by the limited resources, the challenges and requirements for securing assets is equally important for both the developed and non-resource-constrained settings and for the developing and under-developed resource-constrained settings. Therefore, this project seek to develop a kind of a lightweight solution for such resource-constrained environments.

This project seeks to implement face recognition system that only records short videos when an intrusion is detected. This minimizes the amount of memory and transmission bandwidth requirements. This will be implemented using python-based openCV and any other suitable libraries/packages.

### Technologies
The following technologies may used to implement the system built for this project as shown in Figure 1
- Wired Camera
- Microprocessor
- Internet
  
For the purpose of demonstration and testing, this project was tested on a laptop. Thus, relied on the system's camera and computer processor.
![Surveillance System Architecture!](/architecture.drawio.png "Architecture")

### Libraries/Packages
The following python libraries were used for this project
- Opencv
- face_recognition
- facenet-pytorch [3]
- mtcnn [4]
- numpy
- Pillow
- dlib
### Installing Dependencies
From the project's main directory, run the following command:

  ``` pip install -r requirements.txt ```

### Running the System
To get help how to run the system, execute python file as follows:

``` {python|python3} SurveillanceApp.py --help ```

To add an authorized user, execute the following:

 ``` {python|python3} SurveillanceApp.py --name <UserName> ``` 
 
For example,

 ``` python3 SurveillanceApp.py --name Babangida ``` 

This opens a the system's camera and snaps user's image, creates a directory and stores the image to the directory.
Then the face encoding is recreated and stored for the face recognition app.

To start the surveillance system, execute the following:

``` {python|python3} SurveillanceApp.py --start -m <hog|facenet> ```

For example, run:

``` python3 SurveillanceApp.py --start -m facenet ```

Images of authorized persons/users may be added manually by adding them to the directory:

``` <projectDirectory/data/UserName/image.jpg> ```

In such a case, user needs to rerun the encoding (as described above) so as to update the system.

### Challenges
This project was tested on a computer and executed successfully. It showed successful detection of faces from images from opencv camera feed, and recognition of known (previously encoded/embedded) faces. It also was able to send email once the image is made of all unknown faces.

It was obeserved that the facenet model, which used the MTCNN and InceptionResnetV1(pretrained='vggface2') seem to perform better than the HOG-based model from face_recognition package.

Although a reasonable level of success was achieved in this project,it cannot claim to be very effective as it was observed that images containing faces of persns previously encoded/embedded had the probability of mis-recognition. That is, the face may be detected but not recognized as the face of an autorized person. This may be as a result of facial expression, illumination, orientation or partial occlusion of the face.

### Future Enhancements
Following the challenge of mis-recognition as described above, this presents opportunities for enhancements. For example, Images of authorizes persons may be taken at different orientations, with different illumination and facial expressions such as:

``` ['Neutral', 'Happiness-Smile', 'Anger-Frown', 'Sadness', 'Surprise', 'Disgust', 'Fear'] ```

or more. However, we need to note that in such a case, the computational requirement of the system will become higher. Thus, it would become a kind of an optimization problem.

### References
[1] Calba, C., Goutard, F.L., Hoinville, L. et al. Surveillance systems evaluation: a systematic review of the existing approaches. BMC Public Health 15, 448 (2015). https://doi.org/10.1186/s12889-015-1791-5

[2] Gouglidis, A., Green, B., Hutchison, D. et al. Surveillance and security: protecting electricity utilities and other critical infrastructures. Energy Inform 1, 15 (2018). https://doi.org/10.1186/s42162-018-0019-1

[3] F. Schroff, D. Kalenichenko and J. Philbin, "FaceNet: A unified embedding for face recognition and clustering," 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Boston, MA, USA, 2015, pp. 815-823, doi: 10.1109/CVPR.2015.7298682.

[4] Zhang, K., Zhang, Z., Li, Z., and Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters, 23(10):1499–1503.
