import pandas as pd

data = pd.read_csv("../train-labels/train.csv") 

result = data.where(data["id_code"]=="HEPG2-01_1_B06", inplace=False).dropna()

print("RESULT: ", result)
 
sirna = result.iloc[0].loc["sirna"]

print("SIRNA: ", sirna)

