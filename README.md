Donwload Model [MLDN-MODEL]((https://huggingface.co/prajwalkumarprof/MLDN-fine-tuned).

VISIT MLDN-SOURCE (https://github.com/prajwalkumarprof/MLDN-OCR)
## MLDN-OCR   
 

```python
  
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import os
import glob
from tqdm.auto import tqdm
from urllib.request import urlretrieve
from zipfile import ZipFile
 
 
import numpy as np
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('MLDN-fine-tuned/checkpoint-100')
#trocr_fine_tuned/checkpoint-100

def ocr(image, processor, model):
     
    pixel_values = processor(image, return_tensors='pt').pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text
 
def read_image(image_path):
    
    image = Image.open(image_path).convert('RGB')
    return image

def eval_new_data(data_path=None, num_samples=4, model=None):
    image_paths = glob.glob(data_path)
    for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
        if i == num_samples:
            break
        image = read_image(image_path)
        text = ocr(image, processor, model)
        plt.figure(figsize=(7, 4))
        plt.imshow(image)
        plt.title(text)
        plt.axis('off')
        plt.show()

eval_new_data(
    data_path=os.path.join('images', 'malicious', '*'),
    num_samples=2,
    model=model
)
```
