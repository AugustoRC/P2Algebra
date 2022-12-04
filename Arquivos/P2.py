import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

datas = pd.read_csv("./input/InvestidoresEmProjetosRenunciaFiscalEDITADO2.csv")

# print(datas.DESCR)

datas["ART25"] = datas["ART25"].fillna(0)
datas["ART18"] = datas["ART18"].fillna(0)
datas["ART1A"] = datas["ART1A"].fillna(0)
datas["ART1"] = datas["ART1"].fillna(0)
datas["ART39"] = datas["ART39"].fillna(0)
datas["ART3A"] = datas["ART3A"].fillna(0)
datas["ART3"] = datas["ART3"].fillna(0)
datas["ART41"] = datas["ART41"].fillna(0)
datas["VL_TOTAL_INVESTIDO"] = datas["VL_TOTAL_INVESTIDO"].fillna(0)
dataStrCNPJ = datas["CNPJ_INVESTIDOR"]
dataStrInv = datas["INVESTIDOR"]
del datas["CNPJ_INVESTIDOR"]
del datas["INVESTIDOR"]
del datas["Unnamed: 0"]

print(datas)

df = pd.DataFrame(datas)
df.head()

scaler = StandardScaler()
x_scaled = scaler.fit_transform(df)
x_scaled

pca = PCA(n_components=2, random_state=42)
pca.fit(x_scaled)
x_pca = pca.transform(x_scaled)
x_scaled.shape, x_pca.shape

plt.figure(figsize=(12,8))
plt.scatter(x_pca[:,0], x_pca[:, 1], c = datas.target, cmap= "viridis")
plt.xlabel('Frist Principal Component')
plt.ylabel("Second Principal Component")

pca.explained_variance_ratio_

pca = PCA(n_components=20, random_state=42)
x_pca = pca.fit_transform(x_scaled)
variance = pca.explained_variance_ratio_
plt.bar(x = range(1, len(variance)+1), height=variance, width=0.7)
variance