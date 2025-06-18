from paddleocr import PaddleOCR
import cv2
import numpy as np

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Add 'hi' for Hindi if needed

def extract_text(image):
    # Convert PIL image to OpenCV format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    result = ocr.ocr(image, cls=True)
    text = ""
    for line in result[0]:  # PaddleOCR v2.8.1 format
        text += line[1][0] + " "
    return text.strip()