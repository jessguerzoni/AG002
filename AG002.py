import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


#01 - 02
#baixar dataet e fazer leitura dos dados

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

# --- passos anteriores (conversão e reordenação) ---
df = pd.read_csv("Wholescale costumers.csv")
df["Channel"] = df["Channel"].map({"HoReCa": 0, "Retail": 1}).astype("int64")
df["Region"] = df["Region"].map({"Lisbon": 0, "Oporto": 1, "Other": 2}).astype("int64")
df = df.reindex(columns=["Region", "Fresh", "Milk", "Grocery", "Frozen",
                         "Detergents_Paper", "Delicatessen", "Channel"])

# --- separar características (X) e alvo (y) ---
X = df.drop(columns="Channel")
y = df["Channel"]

# --- dividir em treino e teste ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- 6 e 7: escolher o modelo, treinar e classificar o conjunto de teste ---
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# --- 8: métricas de avaliação ---
print(classification_report(y_test, y_pred, target_names=["HoReCa", "Retail"])) 

# --- 9: classificar dados digitados pelo usuário ---
print("\nInforme os dados de um cliente para classificar:")
regiao = int(input("Region (0 = Lisbon, 1 = Oporto, 2 = Other): "))
fresh = float(input("Fresh: "))
milk = float(input("Milk: "))
grocery = float(input("Grocery: "))
frozen = float(input("Frozen: "))
detergents = float(input("Detergents_Paper: "))
delicatessen = float(input("Delicatessen: "))

novo = pd.DataFrame(
    [[regiao, fresh, milk, grocery, frozen, detergents, delicatessen]],
    columns=X.columns,
)
pred = modelo.predict(novo)[0]
print("Canal de vendas previsto:", "Retail" if pred == 1 else "HoReCa")
