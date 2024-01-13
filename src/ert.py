import cv2
import matplotlib.pyplot as plt
import torch
from skimage import io
import time
import locale
import functools
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import os

# project_dir =  "/home/zuoxy/facetrack/"
# vid_id = "004"
def extract_first_frame(vid_path):
	# data_dir = project_dir + "data/300vw/300VW_Dataset_2015_12_14/"
	# vid_path = data_dir + vid_id + "/vid.avi"
	# img_dir = project_dir + "assets/images/"
	# capture first frame of video
	vidcap = cv2.VideoCapture(vid_path)
	success,image = vidcap.read()
	# resize it to 256 by 256
	image = cv2.resize(image, (256, 256))
	im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	return image, gray

def ert_detect(project_dir, vid_id=4, custom=False):
	predictor_path = project_dir + "models/shape_predictor_68_face_landmarks.dat"
	data_dir = project_dir + "data/300vw/300VW_Dataset_2015_12_14/"
	vid_path = data_dir + vid_id + "/vid.avi"
	img_dir = project_dir + "assets/images/"

	# capture first frame of video, resize it and convert it to grayscale
	image, gray = extract_first_frame(project_dir, vid_id)
	cv2.imwrite(img_dir+"input.jpg", image)
	cv2.imwrite(img_dir+"input_gray.jpg", gray)

	#initialize facial landmark detector 
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(predictor_path)

	# detect faces in the grayscale image
	rects = detector(gray, 1)
	if(len(rects) == 0):print("No face detected")

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		# cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		# # show the face number
		# cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
	cv2.imwrite(img_dir+"detect_ert.jpg", image)

	frame_idx = np.zeros((68, 1))
	points = np.concatenate((frame_idx, shape[:,1].reshape(68,1), shape[:,0].reshape(68,1)), 1).astype(np.int32)
	return points 

# pts = ert_detect(project_dir, vid_id)
# print(pts)
