import layoutparser as lp 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2

class Detect:
    
    def __init__(self, scan_table=True, warmup=None):
        self.ocr_agent = lp.TesseractAgent(languages='eng+ind')
        self.model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})

    def detect_objects(self, image):
        layout = self.model.detect(image)
        text_blocks = lp.Layout([b for b in layout if b.type=='Text'])
        figure_blocks = lp.Layout([b for b in layout if b.type=='Figure'])
        text_blocks = lp.Layout([b for b in text_blocks if not any(b.is_in(b_fig) for b_fig in figure_blocks)])
        
        h, w = image.shape[:2]

        left_interval = lp.Interval(0, w/2*1.05, axis='x').put_on_canvas(image)

        left_blocks = text_blocks.filter_by(left_interval, center=True)
        left_blocks.sort(key = lambda b:b.coordinates[1])

        right_blocks = [b for b in text_blocks if b not in left_blocks]
        right_blocks.sort(key = lambda b:b.coordinates[1])

        text_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])
        data_text = []
        for block in text_blocks:
            segment_image = (block
                            .pad(left=5, right=5, top=5, bottom=5)
                            .crop_image(image))
            text = self.ocr_agent.detect(segment_image)
            data_text.append(text)
        
        # for txt in data_text:
        #     print(txt, end="\n---\n")
        print(int(figure_blocks[0].block.y_1))
        for i, fig in enumerate(figure_blocks):
            cropped_image = image[int(fig.block.y_1):int(fig.block.y_2), int(fig.block.x_1):int(fig.block.x_2)]
            cv2.imwrite(f"/home/langchain/from_github/langchain/resource/{i}.png", cropped_image)