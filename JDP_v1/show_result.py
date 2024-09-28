import pandas as pd

def process_csv(file_path):
 
    data = pd.read_csv(file_path)
    
    
    print(f"csv name: {file_path}")
    
   
    if data.shape[1] < 2:
        raise ValueError("atleast two columns")

    first_col = data.iloc[:, 0]
    count_1 = (first_col == 1).sum()

    
    second_col = data.iloc[:, 1]
    sum_second_col = second_col.sum()

   
    return [count_1, sum_second_col]


def main():
    base_path = 'path/benchmark/audio_result/qwen2_at/female/'
    output_data = pd.DataFrame()

   
    for i in range(1, 11):
        file_path = f'{base_path}{i}.csv'
        
        
        result = process_csv(file_path)
        
        
        output_data[f'file_{i}_count_1'] = [result[0]]  
        output_data[f'file_{i}_sum'] = [result[1]]     

    
    output_data.to_csv('output.csv', index=False)
    print("save to output.csv")

main()
