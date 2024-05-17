#!/home/babangida/Surveillance/surv/bin/python
# coding: utf-8


#Import packages
from pathlib import Path
import face_recognition
import pickle
from collections import Counter
from PIL import Image, ImageDraw
import cv2 
import datetime
from sendEmail import SendEmail
import glob

#Define constants
DEFAULT_ENCODINGS_PATH = Path("data/output/encodings.pkl")
BOUNDING_BOX_COLOR = (0, 0, 255) #"blue"
TEXT_COLOR = (255, 255, 255) #"white"

class HogRecognizer:
    #Define a constructor
    def __init__(self):
        #Create basic directories to store images and encodings
        Path("data/training").mkdir(exist_ok=True)
        Path("data/output").mkdir(exist_ok=True)
        Path("data/validation").mkdir(exist_ok=True)
        Path("data/images").mkdir(exist_ok=True)
        Path("data/videos").mkdir(exist_ok=True)
        self.images_path = "data/images"
        self.videos_path = "data/videos"
        
        

    def EncodeKnownFaces(self, model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
        names = []
        encodings = []
        for filepath in Path("data/training").glob("*/*"):
            name = filepath.parent.name
            image = face_recognition.load_image_file(filepath)
            
            face_locations = face_recognition.face_locations(image, model=model)
            face_encodings = face_recognition.face_encodings(image, face_locations)
    
            for encoding in face_encodings:
                names.append(name)
                encodings.append(encoding)
                
        name_encodings = {"names": names, "encodings": encodings}
        with encodings_location.open(mode="wb") as f:
            pickle.dump(name_encodings, f)

    #def RecognizeFaces(self, image_location: str, model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH,) -> None:
    def RecognizeFaces(self, model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH,)-> None:
        with encodings_location.open(mode="rb") as f:
            loaded_encodings = pickle.load(f)
        while True:
			# Initialize the webcam
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                break
        known_names = loaded_encodings["names"]
        #print(known_names)
        rec_vid = False
        min_unknown_frames = 0 #minimum number of frames with unknown faces before email is sent and video recorded
        num_frames_vid = 0
		#loop until the user presses 'SPACE Key' or the camera is closed
        while cap.isOpened(): # read a frame from the camera
			# Capture the image
            ret, frame = cap.read()

			# Check if the image is valid
            if ret:
                #input_image = face_recognition.load_image_file(image_location)
                input_image = frame
                input_face_locations = face_recognition.face_locations(input_image, model=model)
                input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)
                
                # Loop over each face found in the frame
                for (top, right, bottom, left), unknown_encoding in zip(input_face_locations, input_face_encodings):
                    # See if the face is a match for the known faces
                    matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding)
                    name = "Unknown"
                    #print(known_names)

                    # If a match was found, use the first one
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_names[first_match_index]
                        min_unknown_frames = 0 #Known face detected, thus initialized this variable
                    else:
                        #No known face. Thus, increment frame counter
                        min_unknown_frames += 1

                    if min_unknown_frames >= 1:
                        #No known face, thus, suspected intrusion-record video-clip and send Email.
                        rec_vid = True

                    if rec_vid:
                        if num_frames_vid <= 50: #create 200 frames images for creation of videos
                            filename = f"{self.images_path + '/img_' + str(num_frames_vid)}.jpg"
                            cv2.imwrite(filename, frame)
                            num_frames_vid += 1
                        else:
                            # Send email, create videos from the saved image frames and delete.
                            print('Suspected intrusion detected. Video recorded and being processed.')
                            rec_vid = False
                            num_frames_vid = 0
                            img_array = []
                            size = (1,1)
                            for filename in glob.glob(self.images_path +"/*.jpg"):
                                img = cv2.imread(filename)
                                height, width, layers = img.shape
                                size = (width,height)
                                img_array.append(img)
                            
                            vidName = str(datetime.datetime.now())
                            vidName = vidName.replace(':', '-')
                            vidName = vidName.replace(' ', '-')
                            vidName = vidName.replace('.', '-') 
                            filename = f"{self.videos_path + '/video-' + vidName}.avi"
                            #print(filename)
                            out = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'DIVX'), 10, size)

                            for i in range(len(img_array)):
                                out.write(img_array[i])

                            out.release()
                            sndMail = SendEmail()
                            subject='Suspected Intrusion ' #+ vidName
                            message = 'Suspected intrusion detected. Watch the attached video and act accordingly.'
                            sndMail.SendMessage(subject, message, filename)
                            print('Suspected intrusion detected. Recorded video processed and sent.')
                            #delete images
                            for filename in glob.glob(self.images_path +"/*.jpg"):
                                os.remove (filename)
                                
                                                
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                    # Draw a label with the name below the face
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font,0.5, (255, 255, 255), 1)

                # Display the resulting image
                cv2.imshow('Resource Constrained Surveillance', frame)
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def FacesInImages(self, unknown_encoding, loaded_encodings):
        boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding)
        votes = Counter(
            name
            for match, name in zip(boolean_matches, loaded_encodings["names"])
            if match
        )
        if votes:
            return votes.most_common(1)[0][0]

    def DisplayFaces(self, draw, bounding_box, name):
        top, right, bottom, left = bounding_box
        draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
        text_left, text_top, text_right, text_bottom = draw.textbbox(
            (left, bottom), name
        )
        draw.rectangle(
            ((text_left, text_top), (text_right, text_bottom)),
            fill="blue",
            outline="blue",
        )
        draw.text(
            (text_left, text_top),
            name,
            fill="white",
        )
    def DisplayBoundingBox(self, image, bndBox, name):
       
        # Draw the rectangle on the image
        y1, x2, y2, x1 = bndBox
        cv2.rectangle(image, (x1, y1), (x2, y2), color=BOUNDING_BOX_COLOR, thickness=2)
        # Draw the name of the bounding box above the rectangle
        cv2.putText(
            img=image,
            text=name,
            org=(x1, y1 - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=TEXT_COLOR,
            thickness=2
        )
        # Display the image
        
        cv2.imshow("Face Recognition with Name", image)
        k = cv2.waitKey(1)


		
    def Validate(self, model: str = "hog"):
        for filepath in Path("data/validation").rglob("*"):
            if filepath.is_file():
                RecognizeFaces(image_location=str(filepath.absolute()), model=model)



# In[ ]:




