import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "7"  
import csv
from modelscope import (
    snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
)
import torch
model_id = 'path/Qwen-Audio-Chat'
revision = 'master'

model_dir = 'path/Qwen-Audio-Chat'#snapshot_download(model_id, revision=revision)

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
if not hasattr(tokenizer, 'model_dir'):
    tokenizer.model_dir = model_dir

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cpu", trust_remote_code=True).eval()
# use cuda device
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda", trust_remote_code=True).eval()

def query(audio,text):
# 1st dialogue turn
    query = tokenizer.from_list_format([
        {'audio': audio}, # Either a local path or an url
        {'text': text},
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
                audio = f"/home/beihang/yzh/benchmark/minibench_final/audio_data_male/{folder_number}/{i+1}.wav"
                result = query(audio,text)
                writer.writerow([result])


def main():
    for folder_number in range(1, 11):  
        input_csv = f'path/benchmark/minibench_final/text/{folder_number}.csv'
        output_csv = f'path/benchmark/audio_ans/qwen/male/{folder_number}.csv'
        
        process_csv(input_csv, output_csv, folder_number)

main()

