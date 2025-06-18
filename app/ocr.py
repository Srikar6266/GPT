from paddleocr import PaddleOCR
import cv2
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize PaddleOCR
try:
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)  # Add 'hi' for Hindi if needed
    logger.info("PaddleOCR initialized with GPU support")
except Exception as e:
    logger.warning(f"GPU initialization failed: {e}. Falling back to CPU")
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

def extract_text(image):
    try:
        # Convert PIL image to OpenCV format
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        result = ocr.ocr(image, cls=True)
        text = ""
        for line in result[0]:  # PaddleOCR v2.8.1 format
            text += line[1][0] + " "
        logger.info("Text extracted successfully")
        return text.strip()
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        return ""