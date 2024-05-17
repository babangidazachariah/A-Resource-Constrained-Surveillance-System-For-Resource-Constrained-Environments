# Import the libraries
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
import os
from sendEmail import SendEmail
import glob
import datetime

# Define a class that can perform face recognition
class FaceNetRecognizer:
    # Initialize the face detector and the face embedding model
    def __init__(self):
        self.mtcnn = MTCNN()
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.images_path = "data/images"
        self.videos_path = "data/videos"
        
    # Define a function that creates embeddings of images of known faces and returns the embeddings and the detected faces bounding boxes list
    def create_embeddings(self, known_dir):
        # Load the images of known faces and convert them to tensors
        known_faces = os.listdir(known_dir) # Get the names of the subdirectories
        known_tensors = []
        known_names = []
        for face in known_faces:
            face_dir = os.path.join(known_dir, face) # Get the path of the subdirectory
            face_img = os.listdir(face_dir)[0] # Get the first image in the subdirectory
            face_path = os.path.join(face_dir, face_img) # Get the path of the image
            img = cv2.imread(face_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = self.mtcnn(img)
            known_tensors.append(tensor)
            known_names.append(face) # Store the name of the face

        # Embed the known faces using the resnet model
        known_embeddings = []
        for tensor in known_tensors:
            embedding = self.resnet(tensor.unsqueeze(0))
            known_embeddings.append(embedding)

        # Return the embeddings and the names of the known faces
        return known_embeddings, known_names

    # Define a function that recognizes the faces in an opencv image and returns the recognition results and the annotated image
    def recognize_faces(self, opencv_img, known_embeddings, known_names):
        # Convert the opencv image to RGB
        opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)

        # Detect and crop the faces from the opencv image
        opencv_faces, opencv_probs = self.mtcnn.detect(opencv_img)
        # Embed the opencv faces using the resnet model
        opencv_embeddings = []
        opencv_crops = [] # A list to store the cropped faces
        opencv_coords = [] # A list to store the coordinates of the faces
        if opencv_faces is not None:
            for i, (x1, y1, x2, y2) in enumerate(opencv_faces):
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                try:
                    crop = opencv_img[y1:y2, x1:x2]
                    tensor = self.mtcnn(crop)
                    embedding = self.resnet(tensor.unsqueeze(0))
                    opencv_embeddings.append(embedding)
                    opencv_crops.append(crop)
                    opencv_coords.append((x1, y1, x2, y2)) # Store the coordinates as a tuple
                    #print('Face found')
                except:
                    return None, None

        else:
            #print('Face Not found')
            return None, None

        
        # Compare the opencv embeddings with the known embeddings using cosine similarity
        threshold = 0.70 # You can adjust this value according to your needs
        recognition_results = [] # A list to store the recognition results
        for i, opencv_embedding in enumerate(opencv_embeddings):
            similarities = []
            for known_embedding in known_embeddings:
                similarity = (opencv_embedding * known_embedding).sum() / (opencv_embedding.norm() * known_embedding.norm())
                similarities.append(similarity.item())
            max_similarity = max(similarities)
            max_index = similarities.index(max_similarity)
            if max_similarity > threshold:
                #print(f'Face {i+1} in the opencv image matches with {known_names[max_index]} with similarity {max_similarity:.2f}')
                recognition_results.append(True)
                # Draw the bounding box and the name on the opencv image
                x1, y1, x2, y2 = opencv_coords[i] # Get the coordinates from the list
                cv2.rectangle(opencv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(opencv_img, known_names[max_index], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                #print(f'Face {i+1} in the opencv image does not match with any of the known faces')
                recognition_results.append(False)
                # Draw the bounding box and the name 'unknown' on the opencv image
                x1, y1, x2, y2 = opencv_coords[i] # Get the coordinates from the list
                cv2.rectangle(opencv_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(opencv_img, 'unknown', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Return the recognition results and the opencv image with annotations
        return recognition_results, opencv_img

    # Define a function that starts an opencv camera and passes frames to the recognize_faces function
    def start_surveillance(self, known_dir):
        # Create the embeddings and the names of the known faces
        known_embeddings, known_names = self.create_embeddings(known_dir)
        min_num_frames = 0
        num_frames_vid = 0
        recVid = False
        # Start the opencv camera
        cap = cv2.VideoCapture(0) # Use 0 for the default camera, or 1, 2, etc. for other cameras
        while True:
            # Read a frame from the camera
            ret, frame = cap.read()
            if ret:
                #print(frame)
                # Recognize the faces in the frame
                results, annotated_frame = self.recognize_faces(frame, known_embeddings, known_names)

                if results is not None:
                    #if False in results:
                    if not any(results):

                        #Unknown face detected.
                        recVid = True
                        
                    else:
                        #No unknown face detected
                        #recVid = True
                        pass
                    # Show the annotated frame
                    if recVid: 
                        if num_frames_vid < 50:
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
                            print('Suspected intrusion detected. Recorded video processed.')
                            #delete images
                            for filename in glob.glob(self.images_path +"/*.jpg"):
                                os.remove (filename)
                    
                    cv2.imshow('Face Recognition', annotated_frame)

                # Wait for a key press to exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break # Exit the loop if the 'q' key is pressed

        # Release the camera and destroy all windows
        cap.release()
        cv2.destroyAllWindows()
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


	
# Test the class with an example
#fr = FaceRecognizer() # Create an instance of the class
#known_dir = 'data/training' # Specify the directory of known faces
#fr.start_surveillance(known_dir) # Call the start_surveillance function
