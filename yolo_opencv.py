############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################


import cv2
import numpy as np
import logging
from datetime import datetime
import dataclasses

logging.basicConfig(filename='logs.txt', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-1s %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')

class results:
    def __init__(self, indices, boxes, class_ids, confidences):
        self.indices = indices
        self.boxes = boxes
        self.class_ids = class_ids
        self.confidences = confidences

    def get_NMS(self):
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
            logging.info(f'- NMS not applied')
            return self

        return results_NMS

    def draw(self, image):

        for box, class_id, confidence in zip(self.boxes, self.class_ids, self.confidences):
            x = round(box[0])
            y = round(box[1])
            w = round(box[2])
            h = round(box[3])
            
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            logging.info(
                f'- Detection - class: {class_id} - confidence: {confidence}')
            logging.info(
                f'- bbox - x: {round(x)} - y: {round(y)} - w: {round(w)} - h: {round(h)}')

        return image

class yolo:
    def __init__(self, input, out, weights, config, classes):
        self.input = input
        self.out = out
        self.weights = weights
        self.config = config
        self.classes = classes
        self.colors = self.colors()
        self.filename = self.input.split('/')[-1]
        self.results = None
        self.search = None
        self.image = None
    
    def __init__(self, args):
        self.input = args.input
        self.out = args.out
        self.model = args.model
        self.weights = args.model + '.weights'
        self.config = args.model + '.cfg'
        self.classes = args.classes

        self.colors = self.colors()
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
    
    def colors(self):
        classes = None

        with open(self.classes, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        return [classes, COLORS]

    def draw_prediction(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        
        #classes, COLORS = self.colors()
        classes, COLORS = self.colors[0], self.colors[1]
        label = str(classes[class_id])
        color = COLORS[class_id]

        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

        cv2.putText(img, label, (x-10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

    def save(self, path = None):
        if path is None:
            path = self.out + '/'
        elif '/' not in path:
            path = self.out + '/' + path

        cv2.imwrite(path, self.image)
        logging.info(f'- Saved image : {path}')

    def detect(self, image = None, search = None):
        
        if search != None:
            self.search = search
        
        if image is None:
            logging.info(f'- Loaded image : {self.input}')
            self.image = cv2.imread(self.input)
        else:
            self.image = image

        Width = self.image.shape[1]
        Height = self.image.shape[0]
        scale = 0.00392

        net = cv2.dnn.readNet(self.weights, self.config)

        blob = cv2.dnn.blobFromImage(
            self.image, scale, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        
        outs = net.forward(self.get_output_layers(net))

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
                
        if self.search is not None : logging.info(f'- Found {self.search} in image')
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        
        return results(indices, boxes, class_ids, confidences)

    def draw(self, results):
        
        try:
            for i in results.indices:
                try:
                    box = results.boxes[i]
                except:
                    i = i[0]
                    box = results.boxes[i]

                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                box = [x, y, w, h]
                class_id = results.class_ids[i]
                confidence = results.confidences[i]

                self.draw_prediction(self.image, results.class_ids[i], results.confidences[i], round(
                    x), round(y), round(x+w), round(y+h))

                logging.info(
                    f'- Detection - class: {class_id} - confidence: {confidence}')
                logging.info(
                    f'- bbox - x: {round(x)} - y: {round(y)} - w: {round(w)} - h: {round(h)}')
        except:
            for box, class_id, confidence in zip(results.boxes, results.class_ids, results.confidences):
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                self.draw_prediction(self.image, class_id, confidence, round(
                    x), round(y), round(x+w), round(y+h))
                logging.info(
                    f'- Detection - class: {class_id} - confidence: {confidence}')
                logging.info(
                    f'- bbox - x: {round(x)} - y: {round(y)} - w: {round(w)} - h: {round(h)}')

        #cv2.imshow("object detection", image)
        #cv2.waitKey()
        
        #cv2.destroyAllWindows()