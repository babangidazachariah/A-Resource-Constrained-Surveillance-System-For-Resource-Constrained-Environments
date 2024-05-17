
# coding: utf-8

#Import Libraries/Packages
import argparse
from AuthorizeMember import AuthorizeMember
from Surveillance import Surveillance
from FaceNetRecognition import FaceRecognizer

parser = argparse.ArgumentParser(description="Resource Constraint Surveillance Using Face Recognition")
parser.add_argument("--name", action="store", help="Add an authorized member")
parser.add_argument("--files", action="store", help="List of paths to new member's images")
parser.add_argument("--start", action="store_true", help="Start Surveillance System by specifyin the --start -m <hog|facenet>")
parser.add_argument("--train", action="store_true", help="Train on input data")
#parser.add_argument("--validate", action="store_true", help="Validate trained model")
#parser.add_argument("--test", action="store_true", help="Test the model with an unknown image")
parser.add_argument("-m", action="store", default="hog", choices=["hog", "facenet"], help="Which model (hog or facenet) to use for Surveillance",)

args = parser.parse_args()

if __name__ == "__main__":
	
	if args.train:
		print("Starting Encoding of previously saved images.")
		surv = Surveillance()
		surv.EncodeKnownFaces()
		print("Done Encoding of previously saved images.")
		#EncodeKnownFaces(model=args.m)
	if args.name or args.files:
		#--name or --files flag is used add
		authmem = None
		stat = False
		if args.name and args.files:
			print('Starting the process of adding new authorized member: ', args.name, " with images files provided")
			authmem = AuthorizeMember(name=args.name, filesPaths=args.files)
			stat = True
		else:
			#If list of files are provided but not the name, that should not be processed
			if args.files and not args.name:
				print("To add new authorized user/images, the name of the user must be given using --name")
			else:
				print('Starting the process of adding new authorized member: ', args.name)
				authmem = AuthorizeMember(name=args.name)
				authmem.Authorize()
				stat = True
		if stat:
				print('Done adding new authorized member: ', args.name)
				print("Starting Encoding of Images")
				surv = Surveillance()
				surv.EncodeKnownFaces()
				print("Done Encoding of Images")
				
	if args.start:
		if args.m == 'hog':
			print("Starting HOG-Based Surveillance System")
			print("To terminate at any point, press 'q' key.")
			surv = Surveillance()
			surv.RecognizeFaces()
			print("Stoping Surveillance System")
		elif args.m == 'facenet':
			print("Starting FaceNet-Based Surveillance System")
			print("To terminate at any point, press 'q' key.")
			fr = FaceRecognizer() # Create an instance of the class
			known_dir = 'data/training' # Specify the directory of known faces
			fr.start_surveillance(known_dir) # Call the start_surveillance function
			print("Stoping FaceNet-Based Surveillance System")
"""
	if args.validate:
		Validate(model=args.m)
	if args.test:
		RecognizeRaces(image_location=args.f, model=args.m)
"""