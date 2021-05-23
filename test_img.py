import os
import glob
import argparse
import cv2
from LaneDetection import LaneDetector

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to input images folder/ file")
args = vars(ap.parse_args())

path = args["input"]
if os.path.isdir(path):
    test_images_1 = glob.glob(os.path.join(path,'*.jpg'))
    test_images_2 = glob.glob(os.path.join(path,'*.png'))
    test_images = test_images_1 + test_images_2
elif os.path.isfile(path):
    test_images = [path]
else:
    print("Error: input path must be a folder or a file.")

for test_img in test_images:
    img = cv2.imread(test_img)
    ld = LaneDetector()

    out_img = ld.detect_lanes(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    img = cv2.addWeighted(img,1.0,out_img,0.2,0.0)
    cv2.imshow('Lanes',img)
    if cv2.waitKey(0)==ord('q'):
        break
cv2.destroyAllWindows()