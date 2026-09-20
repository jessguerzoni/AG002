import pandas as pd

#01 - 02
#fazer leitura dos dados

df=pd.read_csv('Wholescale costumers.csv',delimiter=',')


#03

#Converter os valores para numeros inteiros

#convertendo os dados
df = df.replace({"Channel": {"HoReCa": 0, "Retail": 1},
"Region": {"Lisbon": 0, "Oporto": 1, "Other": 2},
})

df[["Channel", "Region"]] = df[["Channel", "Region"]].astype("int64")

print(df[["Channel", "Region"]].dtypes)
print(df["Channel"].value_counts())
print(df["Region"].value_counts())

#04 - Reordenar as colunas

new_order = ["Region", "Fresh", "Milk", "Grocery", "Frozen","Detergents_Paper", "Delicatessen", "Channel"]

df = df.reindex(columns=new_order)
print(list(df.columns)) 
