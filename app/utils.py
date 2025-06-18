import cv2
from PIL import Image
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_image(image):
    try:
        # Convert PIL to OpenCV
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # Enhance contrast and remove noise
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        denoised = cv2.fastNlMeansDenoising(enhanced)
        logger.info("Image preprocessed")
        return Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB))
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise