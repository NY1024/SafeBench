import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "2" 
import csv
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('path/models/MiniCPM-V-2_6', trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('path/models/MiniCPM-V-2_6', trust_remote_code=True)

def chat(img,text):
    image = Image.open(img).convert('RGB')
    #question = 'This image is closely related to the question'+text+'Please understand and analyze it based on the image and the question, and give the answer to the question'
    question = 'This image is closely related to the question'+text+'Please understand and analyze it based on the image and the question, and give the answer to the question'
    
    
    msgs = [{'role': 'user', 'content': [image, question]}]

    res = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer
    )
    return res


def process_csv(input_csv, output_csv, folder_number):
    with open(input_csv, 'r', encoding='utf-8') as infile, open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for i, row in enumerate(reader):
            if row:  
                text = row[0]
                print('-----')
                print(i)
                print('-----')
                #img = f"path/benchmark_data/mini_bench/{folder_number}/{i+1}.png"
                img = f"path/benchmark_data/high_img_gene/{folder_number}/{i+1}.png"
                result = chat(img,text)
                writer.writerow([result])

def main():
    for folder_number in range(1,24):  # 从 1 到 23
        input_csv = f'path/benchmark_data/text/{folder_number}.csv'
        output_csv = f'path/benchmark_ans/opt/minicpm_opt/{folder_number}.csv'
        
        process_csv(input_csv, output_csv, folder_number)

main()