'''
Author:     Gautam Kantesariya
Created:    20/04/2022

'''

from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import pickle
import cv2

args = {

    "model": "model\\V2\\violence_model.h5", # Path of model
    "label-bin": "model\\V2\\lb.pickle",  # Path of label-bin
    "input": "input\\demo.webm",  # enter "camera" for using camera as input
    "output": "output\\Output1.avi", # output file path and name
    "size": 64

}

print("[INFO] loading model and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label-bin"], "rb").read())

mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])

vpath = args["input"]
if (args.get('input') == 'camera' ):
	vpath = 0
vs = cv2.VideoCapture(vpath)

writer = None
(W, H) = (None, None)

while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # clone the output frame, then convert it from BGR to RGB ordering, resize the frame to a fixed 224x224, and then perform mean subtraction
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224)).astype("float32")
    frame -= mean

    # make predictions on the frame and then update the predictions queue
    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    Q.append(preds)

    # perform prediction averaging over the current history of previous predictions
    results = np.array(Q).mean(axis=0)
    i = 1
    label = lb.classes_[i]

    # draw the activity on the output frame to show probability of frame
    prob = results[i]*100

    text_color = (0, 255, 0)  # default : green

    if prob > 70:  # Violence prob
        text_color = (0, 0, 255)  # red
        lable = 'Violence'
    elif prob > 50:
        label = 'Abnormal'
        text_color = (0, 255, 255)  # yellow
    else:
        label = 'Normal'

    text = "State : {:8} ({:3.2f}%)".format(label, prob)
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(output, text, (15, 40), FONT, 0.5, text_color, 2)

    # plot graph over background image
    output = cv2.rectangle(
        output, (15, 50), (35+int(prob)*4, 60), text_color, -1)

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

    # write output to device
    writer.write(output)

    # displays output frame
    cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

print("[INFO] cleaning up...")
writer.release()
vs.release()