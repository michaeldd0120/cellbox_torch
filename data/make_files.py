import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np

adata = sc.read_h5ad('processed_data.h5ad')
viabilities = np.array(adata.obs['viabilities'])
viabilities_vector = viabilities.transpose()
viabilities_vector = viabilities_vector.reshape(-1, 1)


# Convert the list to a pandas DataFrame
df = pd.DataFrame(adata.var)
filename = 'node_index.csv'
df.to_csv(filename, index=False)


data = np.concatenate((np.array(adata.X), viabilities_vector), axis=1)
np.savetxt('expr.csv', data, delimiter=',', fmt='%f')

pert_labels = np.identity(93)
padding = np.zeros((93, 3507))
bang = np.concatenate((pert_labels, padding), axis=1)
np.savetxt('pert.csv', bang, delimiter=',', fmt='%f')
