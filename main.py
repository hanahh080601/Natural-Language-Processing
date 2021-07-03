import pandas as pd
import os

raw_dir = './Train_Full'
csv_dir = './CSV'

def read_data_from_dir(dir, label=None):
    if label is None:
        label = dir.split("/")[-1]
        #convert label to snake_case
        label = "_".join(label.lower().split())
        data = []
        file_paths = os.listdir(dir)
        if file_paths and len(file_paths):
            for file in file_paths:
                with open(f"{dir}/{file}", mode='rb') as f:
                    text = f.read()
                    data.append(text.decode('utf16').strip())
        return label, data

def save_to_csv(label, data):
    df = pd.DataFrame(data, columns=["Text"])
    df.to_csv(f"{csv_dir}/{label}.csv", index=False)

if __name__ == "__main__":
    folder_dirs = os.listdir(raw_dir)
    print(folder_dirs)
    for dir in folder_dirs:
        label, data = read_data_from_dir(f"{raw_dir}/{dir}")
        print(label)
        print("len data: ", len(data))
        save_to_csv(label, data)
    print(folder_dirs)



