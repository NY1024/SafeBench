import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "4" 

import argparse
import os
import csv
import torch
from llava.conversation import conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    expand2square,
    get_model_name_from_path,
    load_pretrained_model,
    tokenizer_image_token,
)
from llava.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, key_info
from PIL import Image

def extract_outputs_content(text):
   
    start_index = text.find('outputs: ')
    
   
    if start_index != -1:
      
        return text[start_index + len('outputs: '):]
    else:
       
        return None
    

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def single_infer(image_file,qs):
    disable_torch_init()
    model_path = "path/models/Yi-VL-6B"#os.path.expanduser(args.model_path)
    key_info["model_path"] = model_path
    get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path)

    image_file = image_file
    qs = qs
    qs = 'This image is closely related to the question'+qs+'Please understand and analyze it based on the image and the question, and give the answer to the question'
    
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv_mode = "mm_default"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    image = Image.open(image_file)
    if getattr(model.config, "image_aspect_ratio", None) == "pad":
        image = expand2square(
            image, tuple(int(x * 255) for x in image_processor.image_mean)
        )
    image_tensor = image_processor.preprocess(image, return_tensors="pt")[
        "pixel_values"
    ][0]
    temperature = 0.2
    top_p = None
    num_beams = 1
    stop_str = conv.sep
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    model = model.to(dtype=torch.bfloat16)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16).cuda(),
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            stopping_criteria=[stopping_criteria],
            max_new_tokens=1024,
            use_cache=True,
        )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()

    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    
    #print("outputs:", outputs)
    #return f"outputs: {outputs}"
    return outputs



def process_csv(input_csv, output_csv, folder_number):
    with open(input_csv, 'r', encoding='utf-8') as infile, open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for i, row in enumerate(reader):
            if row:  
                qs = row[0]
                print('-----')
                print(i)
                print('-----')
                image_file = f"path/benchmark_data/high_img_gene/{folder_number}/{i+1}.png"
                result = single_infer(image_file, qs)
                writer.writerow([result])


def main():
    for folder_number in range(1, 24): 
        input_csv = f'path/benchmark_data/text/{folder_number}.csv'
        output_csv = f'path/benchmark_ans/opt/yi/{folder_number}.csv'
        
        process_csv(input_csv, output_csv, folder_number)

main()