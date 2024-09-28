import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "6" 
import torch
import csv
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("path/models/glm-4v-9b", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    "path/models/glm-4v-9b",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()

def chat_with_mllm(query,img):
    query = 'This image is closely related to the question'+query+'Please understand and analyze it based on the image and the question, and give the answer to the question'
    
    image = Image.open(img).convert('RGB')
    inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": query}],
                                        add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                        return_dict=True)  # chat mode

    inputs = inputs.to(device)


    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        return tokenizer.decode(outputs[0])


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
                img = f"path/benchmark_data/high_img_gene/{folder_number}/{i+1}.png"
                result = chat_with_mllm(query,img)
                writer.writerow([result])

def main():
    for folder_number in range(1, 24):  # 从 1 到 23
        input_csv = f'path/benchmark_data/text/{folder_number}.csv'
        output_csv = f'path/benchmark_ans/opt/glm/{folder_number}.csv'
        
        process_csv(input_csv, output_csv, folder_number)


main()

