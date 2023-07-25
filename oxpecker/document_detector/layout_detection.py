import pytesseract
from PIL import Image
import numpy as np
import os
from time import time
import torch
path_this = os.path.dirname(os.path.abspath(__file__))

class LayoutDetector:
    def __init__(self):
        print(os.path.join('layout_detector', 'yolo_model', 'layout_detector.pt'))
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path=os.path.join('oxpecker/document_detector/layout_detector', 'yolo_model', 'layout_detector.pt'), device='cpu', _verbose=False)
        self.model.conf = 0.2

        with open(os.path.join(path_this,"layout_detector", "classes", "classes.txt"), "r") as f:
            self.class_names = [line.strip() for line in f]

    def sort_boxes(self, boxes):
        centers = []
        for box in boxes:
            x = box[1][0]
            y = box[1][1]
            centers.append((x, y))

        # Sort the boxes based on their y coordinate first, then their x coordinate
        sorted_indices = np.lexsort((np.array(centers)[:, 1], np.array(centers)[:, 0]))

        sorted_boxes = [boxes[i] for i in sorted_indices]

        return sorted_boxes

    def table_detector(self, image, THRESHOLD_PROBA):
        model = self.model_table
        model.overrides['conf'] = THRESHOLD_PROBA
        model.overrides['iou'] = 0.45
        model.overrides['agnostic_nms'] = False
        model.overrides['max_det'] = 1000
        with torch.no_grad():
            outputs = model.predict(image)

        probas = outputs[0].probs


        return (model, probas, outputs[0].boxes)
    
    def detect_objects(self, img):
        results = self.model(img)
        THRESHOLD = 0.8
        TSR_th = 0.8
        padd_top = 60
        padd_left = 20
        padd_right = 20
        padd_bottom = 20

        
        result_box = results.pandas().xyxy[0]
        # print(result_box)
        text = result_box[result_box["name"] == "text"]
        # text = [json.loads(result) for result in result_box if json.loads(result)["name"] == "text"]
        table = []
        
        boxes = []
        for i, t in text.iterrows():
            x, y, x1, y1 = int(t["xmin"]), int(t["ymin"]), int(t["xmax"]), int(t["ymax"])
            cropped_image = img[y:y1, x:x1]
            pil_img = Image.fromarray(cropped_image)
            extract = pytesseract.image_to_string(
                pil_img, lang='ind', config='--oem 3 --psm 1')
            boxes.append([extract, (x, y, x1, y1)])
        if len(boxes) > 0:
            boxes = self.sort_boxes(boxes)
        
        cleaned_text = ""
        for b in boxes:
            cleaned_text += b[0] + "\n\n"
        clean_list = cleaned_text.split("\n\n")
        joined_list = [i.replace("-\n", "").replace("\n", " ").strip() for i in clean_list]
        cleaned_text = "\n\n".join(joined_list)
        
        return cleaned_text
