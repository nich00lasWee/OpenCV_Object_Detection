# References:
# https://www.youtube.com/watch?v=RFqvTmEFtOE
# https://gist.github.com/aallan/fbdf008cffd1e08a619ad11a02b74fa8#file-coco_labels-txt
# https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/
# Test video source: https://www.youtube.com/watch?v=2xkAZ9h_dwg

import numpy as np
import cv2

frozen_model = 'frozen_inference_graph.pb'
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

# Loading model
model = cv2.dnn_DetectionModel(frozen_model, config_file)

labels = {}
file = 'coco_labels.txt'

# Loading labels from text file
with open(file) as f:
    for line in f:
        (key, val) = line.strip().split(' ', 1)
        labels[int(key)] = val

# Generate random colors up to index of last label
colors = np.random.uniform(0, 255, size=(len(labels) + 10, 3))

# Setup according to config file
model.setInputSize((320, 320))
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))


def image_detection(image_source):
    # Opening image
    image = cv2.imread(image_source)
    image = cv2.resize(image, (800, 600))

    # Only include objects with confidence level greater than threshold
    index, confidence, bbox = model.detect(image, confThreshold=0.5)

    # Raise exception when image cannot be identified
    if (len(index) == 0) & (len(confidence) == 0):
        raise ValueError("Unable to identify image. Please try with other image.")

    # For debug purpose
    # print(index)
    # print(confidence)
    # for i in range(index.shape[0]):
    #    print("{}: {}".format(index[i], confidence[i]))

    # Change index & confidence from 2D to 1D array
    for i, conf, box in zip(index.flatten(), confidence.flatten(), bbox):

        # print(i)  # For debug purpose

        # Get start & end coordinate of the box
        (startX, startY, endX, endY) = box.astype("int")

        # Format label text
        label_text = "{}: {:.0f}%".format(labels[i], 100 * conf)

        # Drawing bounding box
        cv2.rectangle(image, box, colors[i - 1], 2)

        # if box is not out of bound
        if startY - 10 > 10:
            y1, y2, y3 = startY - 25, startY, startY - 8
        # if box is out of bound
        else:
            y1, y2, y3 = startY, startY + 25, startY + 15

        # Draw background for label text
        cv2.rectangle(image, (startX - 1, y1), (startX + len(label_text) * 10, y2),
                      colors[i - 1], -1)

        # Draw label text
        cv2.putText(image, label_text, (startX + 3, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

    while True:
        # Display output
        cv2.imshow("Output", image)

        # Terminate program on Esc key pressed
        if cv2.waitKey(0) & 0xFF == 27:
            break


def video_detection(video_source):
    # Loading video
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise IOError("Unable to load video")

    while True:
        ret, frame = cap.read()

        # 0 = webcam
        if video_source == 0:
            # flip frame vertically
            frame = cv2.flip(frame, 1)

        index, confidence, bbox = model.detect(frame, confThreshold=0.5)

        # Check to ensure frame is not empty
        if len(index) != 0:
            for i, conf, box in zip(index.flatten(), confidence.flatten(), bbox):
                # Check to ensure i is at most the last label index
                if i <= 90:
                    (startX, startY, endX, endY) = box.astype("int")

                    label_text = "{}: {:.0f}%".format(labels[i], 100 * conf)
                    cv2.rectangle(frame, box, colors[i - 1], 2)

                    if startY - 10 > 10:
                        y1, y2, y3 = startY - 25, startY, startY - 8
                    else:
                        y1, y2, y3 = startY, startY + 25, startY + 15

                    cv2.rectangle(frame, box, colors[i - 1], 2)
                    cv2.rectangle(frame, (startX - 1, y1), (startX + len(label_text) * 10, y2),
                                  colors[i - 1], -1)

                    cv2.putText(frame, label_text, (startX + 3, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Object Detection", frame)

        # Terminate loop when video ends
        if not ret:
            print("End of video")
            break

        # Terminate loop on Esc key pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release resource & terminate windows
    cap.release()
    cv2.destroyAllWindows()


# Change as appropriate
source = "resources/city_street_footage.mp4"

#image_detection(source)
video_detection(0)
