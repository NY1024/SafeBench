import os
import csv
from openai import OpenAI
import base64

API_SECRET_KEY = "";
BASE_URL = "" 

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def gpt4ov(query,img_path):
    base64_image = encode_image(img_path)
    client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
    response = client.chat.completions.create(
      model="claude-3-5-sonnet-20240620",
      messages=[
        {
          "role": "user",
          "content": [
            #{"type": "text", "text": query}
            {"type": "text", "text": query},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{base64_image}"}
            }
          ],
        }
      ],
      max_tokens=300,
    )

    return response.choices[0].message.content


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
                imgpath = f"path/benchmark/final_bench/{folder_number}/{i+1}.png"
                try:
                    result = gpt4ov(prompt, imgpath)
                except Exception as e:
                    result = f"Error processing data {imgpath}: {e}"
                writer.writerow([result])


def main():
    for folder_number in range(1, 24):  
        input_csv = f'path/benchmark/final_bench/text/text/{folder_number}.csv'
        output_csv = f'path/benchmark/comeicial_result/claude/{folder_number}.csv'
        
        process_csv(input_csv, output_csv, folder_number)

main()

