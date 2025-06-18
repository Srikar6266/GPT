from paddleocr import PaddleOCR
import cv2
import numpy as np
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize PaddleOCR
try:
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
    logger.info("PaddleOCR ready with GPU, maccha!")
except Exception as e:
    logger.warning(f"GPU failed: {e}. Using CPU")
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

async def extract_text_batch(images):
    try:
        text = ""
        for image in images:
            # Convert PIL to OpenCV
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # Run OCR (PaddleOCR is sync, so we yield control)
            result = await asyncio.to_thread(ocr.ocr, img_cv, cls=True)
            for line in result[0]:
                text += line[1][0] + " "
            logger.info("Page processed")
        return text.strip()
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return ""