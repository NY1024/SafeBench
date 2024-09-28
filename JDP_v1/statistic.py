import os
import csv


import re

def extract_info(row):
   
    judgment_result = re.search(r'\*\*Judgment Result:\*\*\s*(\d+)', row)
    safety_rating = re.search(r'\*\*Safety Rating:\*\*\s*(\d+)', row)
    reasoning = re.search(r'\*\*Reasoning:\*\*\s*(.*)', row, re.DOTALL)
    
   
    judgment_result = judgment_result.group(1).strip() if judgment_result else row
    safety_rating = safety_rating.group(1).strip() if safety_rating else row
    reasoning = reasoning.group(1).strip() if reasoning else row
    
    return [judgment_result, safety_rating, reasoning]



def process_csv(input_file, output_file):
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        
        csv_reader = csv.reader(infile)
        csv_writer = csv.writer(outfile)
        
       
        csv_writer.writerow(['Judgment Result', 'Safety Rating', 'Reasoning'])

        reasoning_buffer = ""
        reasoning_active = False
        
        for row in csv_reader:
            row = row[0]  
            if reasoning_active:
               
                if '**' in row: 
                    csv_writer.writerow([judgment_result, safety_rating, reasoning_buffer.strip()])
                    reasoning_buffer = ""
                    reasoning_active = False
                else:
                   
                    reasoning_buffer += " " + row.strip()
                    continue  

           
            judgment_result, safety_rating, reasoning = extract_info(row)
            
            if reasoning is not None:
                if reasoning:  
                    csv_writer.writerow([judgment_result, safety_rating, reasoning])
                else:
                   
                    reasoning_active = True
                    reasoning_buffer = ""
            else:
                
                csv_writer.writerow([judgment_result, safety_rating, ''])

def main():
    #name_variations = ['bunny','cog','glm','intern','llava_m7','llava_v7','llava_v13_backup','minicpm','phi3','phi35','qwen','qwen2_2b','qwen2_7b','share','yi']
    #name_variations = ['llava_m7','llava_v7','llava_v13','phi35','qwen2_2b','qwen2_7b']
    #name_variations = ['bunny','cog','glm','intern','minicpm','phi3','qwen','share','yi']
    name_variations = ['male','female']
    for name in name_variations:
        
        for folder_number in range(1, 11):
            input_file = f'path/benchmark/audio_judge/qwen2_audio/{name}/{folder_number}.csv'
            output_file = f'path/benchmark/audio_result/qwen2_audio/{name}/{folder_number}.csv'
            
    
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            process_csv(input_file, output_file)

main()
