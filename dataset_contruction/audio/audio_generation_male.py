import os
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda:2" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("path/models/parler-tts-large-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("path/models/parler-tts-large-v1")

def tts(prompt):
    
    description = "A male speaker delivers a commanding speech with an authoritative tone, decisively presenting a performance full of vigor and expressiveness. The recording is of very high quality, with the speaker's voice sounding clear and very close up. The speaking speed is relatively slow."

    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    return audio_arr
    #sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)


import csv

def regenerate_audio(prompt):
   
    new_audio = tts(prompt)  
    return new_audio
def process_csv(input_file, output_dir):
    with open(input_file, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        
       
        for idx, row in enumerate(reader, start=1):
            if row: 
                prompt = row[0]
                audio = tts(prompt)
                print(f"{idx}line is processing")
                
              
                output_file = os.path.join(output_dir, f"{idx}.wav")
                try:
                    sf.write(output_file, audio, model.config.sampling_rate)
                except IndexError:
                    print(f"Error: Audio data shape is incorrect for {idx}.wav. Regenerating audio...")
    
                    audio = regenerate_audio(prompt)  
                    if len(audio.shape) == 1:
                        audio = audio.reshape(-1, 1)
                    sf.write(output_file, audio, model.config.sampling_rate)

def process_multiple_csvs(start, end, input_dir, output_base_dir):
    for i in range(start, end + 1):
        input_file = os.path.join(input_dir, f"{i}.csv")
        output_dir = os.path.join(output_base_dir, str(i))
        
   
        os.makedirs(output_dir, exist_ok=True)
        
        process_csv(input_file, output_dir)


input_directory = 'path/benchmark/final_bench/text/text/'
output_directory = 'path/benchmark/audio/audio_data_man'
process_multiple_csvs(6, 12, input_directory, output_directory)