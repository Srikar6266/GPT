import cv2
from PIL import Image
import numpy as np

def preprocess_image(image):
    # Convert PIL to OpenCV
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Enhance contrast and remove noise
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    denoised = cv2.fastNlMeansDenoising(enhanced)
    return Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB))