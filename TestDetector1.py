import cv2
import numpy as np

Weight = "./YOLO/yolov3-custom_last.weights"
Config = "./YOLO/yolov3-custom.cfg"


def YOLOreader(yoloWeight, yoloConfig):
    net = cv2.dnn.readNet(yoloWeight, yoloConfig)
    classNames = []  # initiliaze list for classes names
    with open("coco.names", "r") as f:
        classNames = f.read().splitlines()

    # read video
    camera = cv2.VideoCapture(
        r"C:\Users\Cesar\Downloads\Video Of People Walking (1).mp4")
    #r"C:\Users\Cesar\Downloads\Video Of People Walking.mp4"

    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    framesInVid = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame_number = 0

    while camera.isOpened():
        start_frame_number += 1  # this allows to skip frames if needed
        camera.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
        cameraRead = camera.read()

        # once video hits last frame, windows close automatically
        if (start_frame_number >= framesInVid):
            break

        else:

            _, img = cameraRead
            height, width, _ = img.shape
            # blob
            blob = cv2.dnn.blobFromImage(
                img, 1 / 255, (416, 416), swapRB=True, crop=False)

            # set input
            net.setInput(blob)
            # get output
            outputLayers = net.getUnconnectedOutLayersNames()
            layerOutputs = net.forward(outputLayers)
            # boxes, confidences
            boxes = []
            confidences = []
            # loop for each layer
            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.8:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])  # creating bounding box
                        confidences.append(float(confidence))
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            font = cv2.FONT_ITALIC
            if len(indexes) > 0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = 'Human Face'
                    confidence = str(round(confidences[i], 2))
                    cv2.rectangle(img, (x, y), (x + w, y + h), 2)
                    cv2.putText(img, label + "" + str(float(confidence)
                                * 100) + "%", (x, y + 20), font, 2, (205, 0, 0), 2)
            # show the video
            cv2.imshow('YOLO Recognition', img,)
            key = cv2.waitKey(1)
            if key == 27:  # Press ESC to quit video
                break

    cv2.destroyAllWindows()  # avoids glitches when pressing ESC
    print((sum(confidences))/len(confidences))  # average confidence rating


YOLOreader(Weight, Config)  # call function
