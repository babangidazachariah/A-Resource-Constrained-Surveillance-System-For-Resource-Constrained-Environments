
# coding: utf-8
"""
		0: 'neutral', 
        1: 'happiness', 
        2: 'surprise', 
        3: 'sadness',
        4: 'anger', 
        5: 'disgust', 
        6: 'fear'
"""

#import the necessary packages
import os
import cv2 
import face_recognition 
import numpy as np

class AuthorizeMember:
	def __init__(self, name='Test', filesPaths=[]):
		self.authorizedName = name
		self.filesPaths = filesPaths
		
		
	def Authorize(self):
		"""
		This function detects new member's face in a camera (if self.filesPath is empty), 
		captures the image and saves it to a directory (self.authorizedName) with the same name as self.authorizedName.
		If self.filesPaths is not empty, meaning user supplied at least one image file path,
		A directory with name self.authorizedName is created and the image(s) is/are copied to the directory.
		"""
		print('Authorize Method Executing', self.authorizedName)
		if len(self.filesPaths) > 0:
			print('Authorize Method\'s if Part  Executing')
			pass
		else:
			print('Authorize Method\'s Else Part  Executing')
			# Create a list of prompts
			#prompts = ['Neutral', 'Happiness-Smile', 'Anger-Frown', 'Sadness', 'Surprise', 'Disgust', 'Fear']
			prompts = ['Neutral']

			# Create a directory to store the images
			directory = "data/training/" + self.authorizedName
			if not os.path.exists(directory):
				os.makedirs(directory)

			cap = None
			# Loop through the prompts
			for prompt in prompts:
				# Display the prompt on the screen
				print(f"Please {prompt} and press SPACE key to take a photo.")
				#loop 100 times or until camera is opened
				while True:
					# Initialize the webcam
					cap = cv2.VideoCapture(0)
					if cap.isOpened():
						break

				#loop until the user presses 'SPACE Key' or the camera is closed
				while cap.isOpened(): # read a frame from the camera

					# Capture the image
					ret, frame = cap.read()

					# Check if the image is valid
					if ret:
						 # Display the image with the prompt
						cv2.putText(frame, prompt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
						cv2.imshow("Image", frame)
						k = cv2.waitKey(1)
						if k%256 == 32: # If SPACE key is pressed, save the image

							# Save the image to the directory
							filename = f"{ self.authorizedName + '_' + prompt}.jpg"
							filepath = os.path.join(directory, filename)
							cv2.imwrite(filepath, frame)
							# Display the image
							cv2.imshow("Image", frame)
							# Wait for a key press
							#cv2.waitKey(0)
							# Release the webcam
							cap.release()
							break

					else:
						# Display an error message
						print("Failed to capture the image.")

			
			# Destroy all windows
			cv2.destroyAllWindows()
		
	
