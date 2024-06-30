import pandas as pd

data_path = "../data/"

data = pd.read_csv(data_path + "manual_saferlhf.csv")
labels = []
count = 0
for i in range(len(data)):
    label = data["value"][i].split("\n")
    label = [l.strip() for l in label]
    for l in label:
        if "/" in l:
            count += 1
            print(data.iloc[i]["value"])

print(count)