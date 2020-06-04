import numpy as np
import cv2
import glob
from pickle import load
from skimage.feature import hog
import argparse
from imutils.video import VideoStream
import time


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help = "Path to video")
args = vars(ap.parse_args())

model = load(open('model_p.pkl', 'rb'))

scaler = load(open('scaler.pkl', 'rb'))

def get_hog_features(img, orient = 9, pix_per_cell = 8, cell_per_block = 2, 
                    feature_vec=True, n = 8100):
    features = hog(img, orientations=orient, 
                   pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block), 
                   transform_sqrt=True, 
                   feature_vector=feature_vec)
    features_n = np.array(features, dtype  = "float32").reshape(1, n)
    features_s =  scaler.transform(np.array(features_n, dtype ="float32"))
    
    return features_n

def transform(img, size = 128):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    re_image = cv2.resize(gray, (size, size))
    return re_image

# features = []
# for image in glob.glob("test/*.jpg"):
#     img = transform(image)
#     features.append(list(get_hog_features(img)))

# x_test = scaler.transform(np.array(features, dtype ="float32"))
# print(model.predict(x_test))

print("Start load video")
if not args.get("video", False):
	vs = VideoStream(src=0).start()
else:
	vs = cv2.VideoCapture(args["video"])

time.sleep(2.0)
count = 0
while True:
    print("load frame: {}".format(count))
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    if frame is None:
        break
    img = transform(frame)
    features = get_hog_features(img)
    label = model.predict(features)
    print(label)
    text = ""
    if(label == 1):
        text = "Using phone"
    else:
        text = "Not using phone"
    print("Predict {}".format(text))
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 2, cv2.LINE_AA, False)
    cv2.imshow("result", frame)
    key = cv2.waitKey(1) & 0xFF
    count += 1
