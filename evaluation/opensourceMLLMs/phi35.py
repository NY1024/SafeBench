import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "6" 
from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 
import csv
model_id = "path/models/Phi-3.5-vision-instruct" 

# Note: set _attn_implementation='eager' if you don't have flash_attn installed
model = AutoModelForCausalLM.from_pretrained(
  model_id, 
  device_map="cuda", 
  trust_remote_code=True, 
  torch_dtype="auto", 
  _attn_implementation='flash_attention_2'    
)

# for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
processor = AutoProcessor.from_pretrained(model_id, 
  trust_remote_code=True, 
  num_crops=4
) 



def chat(img,prompt):
    images = []
    placeholder = ""
    image_path = img
    images.append(Image.open(image_path))
    placeholder += f"<|image_1|>\n"
    # Note: if OOM, you might consider reduce number of frames in this example.
    '''
    for i in range(1,20):
        url = f"https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.jpg" 
        images.append(Image.open(requests.get(url, stream=True).raw))
        placeholder += f"<|image_{i}|>\n"
    '''

    prompt = 'This image is closely related to the question'+prompt+'Please understand and analyze it based on the image and the question, and give the answer to the question'
    
    messages = [
        {"role": "user", "content": placeholder+prompt},
    ]

    prompt = processor.tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
    )

    inputs = processor(prompt, images, return_tensors="pt").to("cuda:0") 

    generation_args = { 
        "max_new_tokens": 1000, 
        "temperature": 0.0, 
        "do_sample": False, 
    } 

    generate_ids = model.generate(**inputs, 
    eos_token_id=processor.tokenizer.eos_token_id, 
    **generation_args
    )

    # remove input tokens 
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False)[0] 

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
        output_csv = f'path/benchmark_ans/opt/phi35_opt/{folder_number}.csv'
        
        process_csv(input_csv, output_csv, folder_number)

main()

