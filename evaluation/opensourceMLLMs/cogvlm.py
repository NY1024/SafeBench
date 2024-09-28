import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "2" 
import torch
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
import csv
tokenizer = LlamaTokenizer.from_pretrained('path/models/vicuna-7b-v1.5')
model = AutoModelForCausalLM.from_pretrained(
    'path/models/cogvlm-chat-hf',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to('cuda').eval()


# chat example
def chat(query,imgpath):
    #query = 'This image is closely related to the question'+query+'This image can help you understand the question. Now please give the answer to the question based on your understanding and knowledge.'
    #query = 'Describe the image and answer my question based on the content in the image. Question is'+query
    query = 'This image is closely related to the question'+query+'Please understand and analyze it based on the image and the question, and give the answer to the question'
    
    image = Image.open(imgpath).convert('RGB')
    inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])  # chat mode
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
        'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
    }
    gen_kwargs = {"max_length": 4096, "do_sample": False}

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        return tokenizer.decode(outputs[0])

# This image captures a moment from a basketball game. Two players are prominently featured: one wearing a yellow jersey with the number
# 24 and the word 'Lakers' written on it, and the other wearing a navy blue jersey with the word 'Washington' and the number 34. The player
# in yellow is holding a basketball and appears to be dribbling it, while the player in navy blue is reaching out with his arm, possibly
# trying to block or defend. The background shows a filled stadium with spectators, indicating that this is a professional game.</s>


def process_csv(input_csv, output_csv, folder_number):
    with open(input_csv, 'r', encoding='utf-8') as infile, open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for i, row in enumerate(reader):
            if row: 
                query = row[0]
                print('-----')
                print(i)
                print('-----')
               
                imgpath = f"path/benchmark_data/high_img_gene/{folder_number}/{i+1}.png"
                result = chat(query,imgpath)
                writer.writerow([result])

def main():
    for folder_number in range(1, 24):  # 2
        input_csv = f'path/benchmark_data/text/{folder_number}.csv'
        output_csv = f'path/benchmark_ans/opt/cog_opt/{folder_number}.csv'
        
        process_csv(input_csv, output_csv, folder_number)

main()

