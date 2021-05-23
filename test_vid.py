import cv2
import argparse
from LaneDetection import LaneDetector

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to input video")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
args = vars(ap.parse_args())

ld = LaneDetector()

test_vid = args["input"]
vs = cv2.VideoCapture(test_vid)

writer = None
width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vs.get(cv2.CAP_PROP_FPS))
if args["output"] != "" and writer is None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(args['output'], fourcc, fps, (width,height))

while True:
    (grabbed, frame) = vs.read()
    if not grabbed:
        break
    out_frame = ld.detect_lanes(cv2.cvtColor(frame.copy(),cv2.COLOR_BGR2RGB))
    frame = cv2.addWeighted(frame, 1.0, out_frame, 0.3, 0.0)
    cv2.imshow('Lanes',frame)
    if writer is not None:
        writer.write(frame)
    if cv2.waitKey(10)==ord('q'):
        break
cv2.destroyAllWindows()