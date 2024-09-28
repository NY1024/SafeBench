import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "5,6" 
from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 
import csv
model_id = "path/models/Phi-3-vision-128k-instruct" 

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2') # use _attn_implementation='eager' to disable flash attention

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 
'''
messages = [ 
    {"role": "user", "content": "<|image_1|>\nWhat is shown in this image?"}, 
    {"role": "assistant", "content": "The chart displays the percentage of respondents who agree with various statements about their preparedness for meetings. It shows five categories: 'Having clear and pre-defined goals for meetings', 'Knowing where to find the information I need for a meeting', 'Understanding my exact role and responsibilities when I'm invited', 'Having tools to manage admin tasks like note-taking or summarization', and 'Having more focus time to sufficiently prepare for meetings'. Each category has an associated bar indicating the level of agreement, measured on a scale from 0% to 100%."}, 
    {"role": "user", "content": "Provide insightful questions to spark discussion."} 
] 
'''

def chat(img,prompt):
    prompt = 'This image is closely related to the question'+prompt+'Please understand and analyze it based on the image and the question, and give the answer to the question'
    
    messages = [ 
        {"role": "user", "content": "<|image_1|>\nWhat is shown in this image?"}, 
        {"role": "assistant", "content": "The chart displays the percentage of respondents who agree with various statements about their preparedness for meetings. It shows five categories: 'Having clear and pre-defined goals for meetings', 'Knowing where to find the information I need for a meeting', 'Understanding my exact role and responsibilities when I'm invited', 'Having tools to manage admin tasks like note-taking or summarization', and 'Having more focus time to sufficiently prepare for meetings'. Each category has an associated bar indicating the level of agreement, measured on a scale from 0% to 100%."}, 
        {"role": "user", "content": prompt} 
    ] 


    image = Image.open(img) 

    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0") 

    generation_args = { 
        "max_new_tokens": 5000, 
        "temperature": 0.0, 
        "do_sample": False, 
    } 

    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 

    # remove input tokens 
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

    return response


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
                writer.writerow([result])

def main():
    for folder_number in range(1, 24):  
        input_csv = f'path/benchmark_data/text/{folder_number}.csv'
        output_csv = f'path/benchmark_ans/opt/phi3_opt/{folder_number}.csv'
        
        process_csv(input_csv, output_csv, folder_number)


main()

