import fitz
from typing import List
from oxpecker.schema.document import Document
from tqdm import tqdm
import numpy as np
import cv2
from document_detector.layout_detection import LayoutDetector

class LayoutExtractor:
    def __init__(self):
        self.ld = LayoutDetector()

    def get_page(self, page):
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3)
        original_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        txt = self.ld.detect_objects(original_image)
        return txt

    def extract(self, file_path) -> List[Document]:
        doc = fitz.open(file_path)
        
        results = []
        
        for i, page in tqdm(enumerate(doc), total=len(doc)):
            result = self.get_page(page)
            results.append(result)
            
        text_result = [
            Document(
                doc_content=page,
                metadata=dict(
                    {
                        "source": file_path,
                        "file_path": file_path,
                        "type": "text",
                        "page_number": i + 1,
                        "total_pages": len(doc),
                        "type_data": "pdf_scanned"
                    },
                    **{
                        k: doc.metadata[k]
                        for k in doc.metadata
                        if type(doc.metadata[k]) in [str, int]
                    },
                ),
            )
            for i, page in enumerate(results)
        ]
        return text_result
