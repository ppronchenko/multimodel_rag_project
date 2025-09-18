from PIL import Image
import cv2
import numpy as np
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Initialize BLIP model for image captioning
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

def analyze_image(image_path):
    """
    Генерирует текстовое описание изображения с помощью модели BLIP.
    
    Параметры:
    image_path (str or numpy.ndarray): Путь к файлу изображения или массив данных изображения.
    
    Возвращает:
    tuple: Объект изображения PIL и сгенерированное описание.
    """
    try:
        if isinstance(image_path, str):
            image_data = Image.open(image_path)
        elif isinstance(image_path, np.ndarray):
            image_data = Image.fromarray(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))

        inputs = blip_processor(image_data, return_tensors="pt")
        output = blip_model.generate(**inputs, max_new_tokens=256)
        image_caption = blip_processor.decode(output[0], skip_special_tokens=True)
        
        return image_data, image_caption

    except Exception as e:
        print(f"Ошибка при обработке изображения {image_path}: {e}")
        return None