import pandas as pd
import os
import json


pav_data_folder = '../datasets/pavlopoulos20/data/CAT_LARGE'

out = '../datasets/formatted_datasets'
i = 0 # so that we have a unique index for both
# Iterate over files in the folder
for filename in os.listdir(pav_data_folder):
    if filename.endswith('.csv'):  # Check if the file is a CSV file
        file_path = os.path.join(pav_data_folder, filename)
        # Read CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)
        # Now you can work with the DataFrame (df) as needed
        print("DataFrame from", filename, ":\n", df)
        with open(os.path.join(out,filename.split('.')[0]+'.jsonl'),'a') as f:
            for _,row in df.iterrows():
                row_dict = {
                    'idx': i,
                    'label': row['label'],
                    'context': row['parent'],
                    'target': row['text']
                }
                i += 1
                json.dump(row_dict,f)
                f.writelines('\n')