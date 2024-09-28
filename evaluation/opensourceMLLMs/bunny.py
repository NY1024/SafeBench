import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "5" 
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
import csv
# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

# set device
device = 'cuda'  # or cpu
torch.set_default_device(device)

# create model
model = AutoModelForCausalLM.from_pretrained(
    'path/models/Bunny-v1_1-Llama-3-8B-V',
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    'path/models/Bunny-v1_1-Llama-3-8B-V',
    trust_remote_code=True)

# text prompt
def chat(prompt,imgpath):
    prompt = 'This image is closely related to the question'+prompt+'Please understand and analyze it based on the image and the question, and give the answer to the question'
    #prompt = 'Describe the image and answer my question based on the content in the image. Question is'+prompt
    
    text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{prompt} ASSISTANT:"
    text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
    input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1][1:], dtype=torch.long).unsqueeze(0).to(device)

    # image, sample images can be found in images folder
    image = Image.open(imgpath)
    image_tensor = model.process_images([image], model.config).to(dtype=model.dtype, device=device)

    # generate
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        max_new_tokens=1024,
        use_cache=True,
        repetition_penalty=1.0 # increase this to avoid chattering
    )[0]

    return tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()


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
                imgpath = f"path/benchmark_data/high_img_gene/{folder_number}/{i+1}.png"
                #imgpath = f"path/benchmark_data/black_image.png"
                result = chat(prompt,imgpath)
                writer.writerow([result])

def main():
    for folder_number in range(1, 24):  # 从 1 到 10
        input_csv = f'path/benchmark_data/text/{folder_number}.csv'
        output_csv = f'path/benchmark_ans/opt/bunny_opt/{folder_number}.csv'
        
        process_csv(input_csv, output_csv, folder_number)

main()

