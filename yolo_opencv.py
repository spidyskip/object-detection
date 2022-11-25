############################################
# Object detection - YOLO - OpenCV
# Author : Antonio  (November 25, 2022)
# Website : https://github.com/spidyskip
############################################

import cv2
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(filename='logs.txt', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-1s : %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')

class yolo:
    def __init__(self, input, model, classes,  out ="out", search = None):
        self.input = input
        self.out = out
        self.model = model
        self.weights = model + '.weights'
        self.config = model + '.cfg'
        self.classes = classes
        self.colors = self.setUpColors()
        self.filename = self.input.split('/')[-1]
        self.net = self.load_model()
        self.results = None
        self.search = search
        self.image = None
    
    def __init__(self, args):
        self.input = args.input
        self.out = args.out
        self.model = args.model
        self.weights = args.model + '.weights'
        self.config = args.model + '.cfg'
        self.classes = args.classes
        self.net = self.load_model()
        self.colors = self.setUpColors()
        self.filename = self.input.split('/')[-1]
        self.results = None
        self.search = args.search
        self.image = None  
    
    def get_class_id(self, class_name):
        try:
            classes = None
            with open(self.classes, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
                class_id = classes.index(class_name)
        except:
            class_id = None
        
        return class_id
    
    def get_class_name(self, class_id):
        classes = None
        with open(self.classes, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes[class_id]

    def get_output_layers(self, net):

        layer_names = net.getLayerNames()
        try:
            output_layers = [layer_names[i - 1]
                            for i in net.getUnconnectedOutLayers()]
        except:
            output_layers = [layer_names[i[0] - 1]
                            for i in net.getUnconnectedOutLayers()]

        return output_layers
    
    def get_classes(self):
        classes = None

        with open(self.classes, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        return classes

    def setUpColors(self):
        classes = self.get_classes()

        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        return COLORS

    def save(self, image, path = None):
        if path is None:
            path = self.out + '/'
        elif '/' not in path:
            path = self.out + '/' + path

        cv2.imwrite(path, image)
        logging.info(f' Saved image : {path}')

    def load_model(self):
        net = cv2.dnn.readNet(self.weights, self.config)
        return net

    def detect(self, image = None, search = None):
        
        if search != None:
            if type(search) == int:
                self.search = self.get_class_name(search)
            else:
                self.search = search
        
        if image is None:
            self.image = cv2.imread(self.input)
        else:
            self.image = image

        Width = self.image.shape[1]
        Height = self.image.shape[0]
        scale = 0.00392

        #net = self.load_model()

        blob = cv2.dnn.blobFromImage(
            self.image, scale, (416, 416), (0, 0, 0), True, crop=False)

        self.net.setInput(blob)
        start = datetime.now()
        outs = self.net.forward(self.get_output_layers(self.net))
        end = datetime.now()
        
        # show timing information on YOLONet
        if len(outs) > 0:
            logging.info(f' {self.model.split("/")[-1]} took {(end - start).total_seconds()}s')

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5 and self.search is None:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
            
                elif confidence > 0.5 and self.search == self.get_class_name(class_id):
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                
        if self.search is not None and self.search == self.get_class_name(class_id):
            logging.info(f' Found {self.search} in image')
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        
        return results(indices, boxes, class_ids, confidences)

class results:
    def __init__(self, indices, boxes, class_ids, confidences):
        self.indices = indices
        self.boxes = boxes
        self.class_ids = class_ids
        self.confidences = confidences

    def get_NMS(self):  # Get result after NMS
        boxes = []
        class_ids = []
        confidences = []
        try:

            for i in self.indices:

                try:
                    box = self.boxes[i]
                except:
                    i = i[0]
                    box = self.boxes[i]

                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                box = [x, y, w, h]
                boxes.append(box)

                class_id = self.class_ids[i]
                class_ids.append(class_id)

                confidence = self.confidences[i]
                confidences.append(confidence)

            results_NMS = results(0,
                                  boxes, class_ids, confidences)
        except Exception as e:
            logging.info(f' NMS not applied')
            return self

        return results_NMS

    def draw(self, img, classes=None, colors=None, NMS=True):  # Draw bounding boxes on the image
        image = img.copy()
        if classes is not None:
            with open(classes, 'r') as f:
                classes = [line.strip() for line in f.readlines()]

            COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        if NMS:
            for i in self.indices:

                try:
                    box = self.boxes[i]
                except:
                    i = i[0]
                    box = self.boxes[i]

                x = round(box[0])
                y = round(box[1])
                w = round(box[2])
                h = round(box[3])
                class_id = self.class_ids[i]
                confidence = self.confidences[i]

                if classes is None:
                    cv2.rectangle(image, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)
                elif colors is None:
                    label = str(classes[class_id])

                    cv2.rectangle(image, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)

                    cv2.putText(image, label, (x-10, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    label = str(classes[class_id])
                    color = colors[class_id]

                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

                    cv2.putText(image, label, (x-10, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                logging.info(
                    f' Detection - class: {class_id} - confidence: {confidence}')
                logging.info(
                    f' Detection - bbox - x: {x} - y: {y} - w: {w} - h: {h}')
        else:
            for box, class_id, confidence in zip(self.boxes, self.class_ids, self.confidences):
                x = round(box[0])
                y = round(box[1])
                w = round(box[2])
                h = round(box[3])

                if classes is None:
                    cv2.rectangle(image, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)
                elif colors == None:
                    label = str(classes[class_id])

                    cv2.rectangle(image, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)

                    cv2.putText(image, label, (x-10, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    label = str(classes[class_id])
                    color = colors[class_id]

                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

                    cv2.putText(image, label, (x-10, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                logging.info(
                    f' Detection - class: {class_id} - confidence: {confidence}')
                logging.info(
                    f' Detection - bbox - x: {round(x)} - y: {round(y)} - w: {round(w)} - h: {round(h)}')

        return image
