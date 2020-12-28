import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from matplotlib import pyplot as plt
# Hierarchical clustering
# preparation of data and features
mydata = pd.read_csv("Liion_comp_528.csv")
# Get the mxrd representation from the data frame
X_ini = mydata.iloc[:,4:]
# known li-ion conductivity values 
y = mydata["log10_cond"]

#show mXRD of In16Li8S32Sn4
# mXRD_sample = mydata[mydata["formula_id"] == "B3Li6O9Y1"]
# mXRD_sample = mXRD_sample.iloc[:,4:].values.ravel()
# plt.plot(mXRD_sample)
# plt.show()

# Examine number of clusters from hierarchical clustering, by examine the variance
# function returning the variance
def myobj(X):
    ncut_vec = [i for i in range(2,11)]
    hc_df = mydata.loc[:,['formula_id','index','log10_cond']]
    for i in ncut_vec: hc_df[str(i)] = np.nan
    Z = linkage(X, 'ward')
    # fig = plt.figure(figsize=(25, 10))
    # dn = dendrogram(Z,leaf_rotation=0,leaf_font_size=10)
    # plt.show()
    cutree = cut_tree(Z, n_clusters=ncut_vec)
    for index, i in enumerate(ncut_vec): hc_df[str(i)] = cutree[:,index]
    hc_sum = pd.DataFrame(np.empty((len(ncut_vec),1)), index = [str(i) for i in ncut_vec], columns = ['label_var'])
    # use the measure of summing both labeled and unlabeled
    for group_meth in ncut_vec: 
        aggcount = hc_df.groupby(str(group_meth))["log10_cond"].count()
        out_stat= pd.DataFrame(np.empty((len(aggcount),3)),columns = ['gp','count','var'])
        hc_df_droped = hc_df.dropna(axis=0)
        aggvar= hc_df_droped.groupby(str(group_meth))["log10_cond"].apply(lambda x:x.var()) # calculate variance
        out_stat['gp'] = aggcount.index
        out_stat['count'] = aggcount
        out_stat['var'] = aggvar.reindex(out_stat.index)  # sort aggvar's index like out_stat
        myout3 = (out_stat["count"] * out_stat['var']).sum(skipna=True)
        hc_sum.loc[str(group_meth),'label_var']=myout3
    return hc_sum

if __name__ == "__main__":
    # print the variance for different number of cuts
    print(myobj(X_ini))
    # cut into 7 clusters, since it gives the largest drop in variance
    Z = linkage(X_ini, 'ward')
    cutree = cut_tree(Z, n_clusters=7)
    
