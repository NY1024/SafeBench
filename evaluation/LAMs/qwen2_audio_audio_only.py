import os 
from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import csv
processor = AutoProcessor.from_pretrained("path/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("path/Qwen2-Audio-7B-Instruct", device_map="auto")

def chat(audio):
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": "path/benchmark/audio/guess_age_gender.wav"},
        ]},
        {"role": "assistant", "content": "Yes, the speaker is female and in her twenties."},
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio},
        ]},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(librosa.load(ele['audio_url'], 
                        sr=processor.feature_extractor.sampling_rate)[0]
                    )

    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    inputs.input_ids = inputs.input_ids.to("cuda")

    generate_ids = model.generate(**inputs, max_length=256)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

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
                audio = f"path/benchmark/minibench_final/audio_data_male/{folder_number}/{i+1}.wav"
                result = chat(audio)
                writer.writerow([result])


def main():
    for folder_number in range(1, 11):  
        input_csv = f'path/benchmark/minibench_final/text/{folder_number}.csv'
        output_csv = f'path/benchmark/audio_ans/qwen2_audio/male/{folder_number}.csv'
        
        process_csv(input_csv, output_csv, folder_number)

main()



