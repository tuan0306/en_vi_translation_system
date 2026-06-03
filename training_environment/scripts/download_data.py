import os
import argparse
from datasets import load_dataset
import pandas as pd

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None)
    args=parser.parse_args()
    
    os.makedirs('training_environment/data/raw', exist_ok=True)
    
    try:
        dataset=load_dataset("ura-hcmut/PhoMT")
        for split in dataset.keys():
            split_name = 'val' if split == 'validation' else split
            processed_data=[]
            count=0
            
            limit=args.max_samples if args.max_samples else len(dataset[split])
            
            for item in dataset[split]:
                if count >= limit:
                    break
                
                src_text=item.get('en',"")
                tgt_text=item.get('vi',"")
                
                if not src_text or not tgt_text:
                    continue
                
                processed_data.append({
                    'en':str(src_text).replace('\n',' ').strip(),
                    'vi':str(tgt_text).replace('\n',' ').strip()
                })
                
                count+=1
            
            df=pd.DataFrame(processed_data)
            
            en_file=f"training_environment/data/raw/{split_name}.en"
            vi_file=f"training_environment/data/raw/{split_name}.vi"
            
            with open(en_file, 'w', encoding='utf-8') as f_en, \
                 open(vi_file, 'w', encoding='utf-8') as f_vi:
                     for _,row in df.iterrows():
                         f_en.write(row['en']+'\n')
                         f_vi.write(row['vi']+'\n')
    
    except Exception as e:
        print(f"Error: {e}")
        
if __name__=="__main__":
    main()