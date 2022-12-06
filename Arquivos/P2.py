import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from sklearn import preprocessing 
from sklearn.decomposition import PCA


data = pd.read_csv("./input/ProjetosRenunciaFiscal1.csv", encoding="utf-8")

dataTitulo = data["TITULO_PROJETO"]

del data["TITULO_PROJETO"]
del data["CNPJ_PROPONENTE"]
del data["Unnamed: 0"]
del data["Unnamed: 2"]
del data["UF_PROPONENTE"]
del data["SITUACAO_REGISTRO"]
del data["RAZAO_SOCIAL_PROPONENTE"]


data["LEI_8313"]= data["LEI_8313"].str.replace(",", ".")
data["ART1"] = data["ART1"].str.replace(",", ".")
data["ART1A"]= data["ART1A"].str.replace(",", ".")
data["ART3"] = data["ART3"].str.replace(",", ".")
data["ART3A"]= data["ART3A"].str.replace(",", ".")
data["ART39"]= data["ART39"].str.replace(",", ".")
data["FUNCINES"] = data["FUNCINES"].str.replace(",", ".")
data["TOTAL_CAPTADO"]= data["TOTAL_CAPTADO"].str.replace(",", ".")


print(data)
print(data.head())
print(data.shape)
scaled_data = preprocessing.scale(data)

dataIndex = object

for gene in data.index:
    dataTitulo = data.loc[gene] 

pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

per_var = np.round(pca.explained_variance_ratio_ * 100, decimals = 1)
labels = ['PC' + str (x) for x in range (1, len(per_var)+1)]

plt.bar(x=range(0, len(per_var)), height=per_var, tick_label = labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()


pca_df = pd.DataFrame(pca_data, index = data, columns = labels)

plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('My PCA Graph')
plt.xlabel('PC1'.format(per_var))
plt.ylabel('PC2'.format(per_var))


plt.show()

loading_scores = pd.Series(pca.components_[0], index = dataTitulo)
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)

top_10_projetos = sorted_loading_scores[0:10].index.values

print(loading_scores[top_10_projetos])