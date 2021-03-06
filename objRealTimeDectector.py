import cv2
import numpy as np


def videoDetector():
    # dnn net read yolo and coco name
    net = cv2.dnn.readNet("./YOLO/yolov3.weights", "./YOLO/yolov3.cfg")
    # classes name
    classes = []
    with open("coco.names", "r") as f:
        classes = f.read().splitlines()
    # read video
    # def generate_frame():
    #camera = cv2.VideoCapture(r"C:\Users\gusta\Downloads\WIN_20220315_08_43_14_Pro.mp4")
    camera = cv2.VideoCapture("test.mp4")
    # camera = cv2.VideoCapture(0)
    lengther = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame_number = 0
    while True:
        try:
            start_frame_number += 10
            camera.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
            success, frame = camera.read()
            if not success:
                if ( start_frame_number >= lengther):
                    break
                else:
                    continue

            else:
                #     ret,buffer=cv2.imencode('.jpg',frame)
                #     frame=buffer.tobytes()
                # yield(b'--frame\r\n'
                #            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                _, img = camera.read()
                height, width, _ = img.shape
                # blob
                blob = cv2.dnn.blobFromImage(
                    img, 1 / 255, (416, 416), swapRB=True, crop=False)

                # set input
                net.setInput(blob)
                # get output
                output_layers = net.getUnconnectedOutLayersNames()
                layer_outputs = net.forward(output_layers)
                # boxes, confidences, class_ids
                boxes = []
                confidences = []
                class_ids = []
                # loop for each layer
                for output in layer_outputs:
                    for detection in output:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                font = cv2.FONT_HERSHEY_PLAIN
                colors = np.random.uniform(0, 255, size=(len(classes), 3))
                if len(indexes) > 0:
                    for i in indexes.flatten():
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        confidence = str(round(confidences[i], 2))
                        color = colors[class_ids[i]]
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(img, label + "" + str(float(confidence)*100) + "%",
                                    (x, y + 20), font, 2, (205, 0, 0), 2)
                        print(label)
                    # read the camera frame
                # show image
                # cv2.imshow('Image',img)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', img)[1].tobytes() + b'\r\n')
                # key = cv2.waitKey(1)
                # if key == 27:
                #     break
        except:
            continue


    cv2.destroyAllWindows()
