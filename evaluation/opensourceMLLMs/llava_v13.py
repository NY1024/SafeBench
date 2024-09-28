import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "1" 
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import csv
def extract_assistant_content(text):
   
    keyword = "ASSISTANT:"
    index = text.find(keyword)
    
    if index != -1:
        
        return text[index + len(keyword):].strip()
    
  
    return text

processor = LlavaNextProcessor.from_pretrained("path/models/llava-v1.6-vicuna-13b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained("path/models/llava-v1.6-vicuna-13b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
model.to("cuda:0")

# prepare image and text prompt, using the appropriate prompt template
def chat(img,prompt):
    prompt = 'This image is closely related to the question'+prompt+'Please understand and analyze it based on the image and the question, and give the answer to the question'
    

    image = Image.open(img)
    # Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image") 
    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=500)

    return processor.decode(output[0], skip_special_tokens=True)



def process_csv(input_csv, output_csv, folder_number):
    with open(input_csv, 'r', encoding='utf-8') as infile, open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for i, row in enumerate(reader):
            if row:  
                prompt = row[0]
                print('-----')
                print(i)
                print('-----')
                img = f"path/benchmark_data/high_img_gene/{folder_number}/{i+1}.png"
                result = chat(img,prompt)
                result = extract_assistant_content(result)
                writer.writerow([result])

def main():
    for folder_number in range(1, 24):  # 从 1 到 23
        input_csv = f'path/benchmark_data/text/{folder_number}.csv'
        output_csv = f'path/benchmark_ans/opt/llava_v13/{folder_number}.csv'
        
        process_csv(input_csv, output_csv, folder_number)

# 执行主函数
main()

