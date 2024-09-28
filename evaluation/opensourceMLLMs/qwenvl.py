
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import csv

tokenizer = AutoTokenizer.from_pretrained("path/models/Qwen-VL-Chat", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("path/models/Qwen-VL-Chat", device_map="cuda:4", trust_remote_code=True).eval()

# Specify hyperparameters for generation (No need to do this if you are using transformers>=4.32.0)
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# 1st dialogue turn
def chat(imgpath,prompt):
    prompt = 'This image is closely related to the question'+prompt+'Please understand and analyze it based on the image and the question, and give the answer to the question'
    
    query = tokenizer.from_list_format([
        {'image': imgpath},
        {'text': prompt},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
    return response

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
                imgpath = f"path/benchmark_data/mini_bench/{folder_number}/{i+1}.png"
                result = chat(imgpath,text)
                writer.writerow([result])


def main():
    for folder_number in range(1, 11): 
        input_csv = f'path/benchmark_data/minitext/{folder_number}.csv'
        output_csv = f'path/benchmark_ans/img_quality/qwen_opt/{folder_number}.csv'
        
        process_csv(input_csv, output_csv, folder_number)

main()

