import pandas as pd
import networkx as nx
import numpy as np
import os

work_dr='/Users/sdz/PycharmProjects/OPRL7/input_cptac_6_4_na_exp'

f1 = pd.read_csv(os.path.join(work_dr,'profiles.csv'), sep=',',index_col=0)
f2 = pd.read_csv('/Users/sdz/Desktop/桌面/micheal zhang/networks/sapiensIntPathGenePairs', header=None, sep='\t', index_col=None)
# f3 = pd.read_csv('/Users/sdz/Desktop/micheal zhang/networks/musculusIntPathGenePairs', header=None, sep='\t', index_col=None)
# # f3 = pd.read_csv('/Users/sdz/Downloads/sapiens/sapiensIntPathGenes', header=None, sep='\t', index_col=None)
# f4 = pd.read_csv('/Users/sdz/Desktop/micheal zhang/networks/musculusIntPathGenes', header=None, sep='\t', index_col=None)
# f5 = pd.read_csv('/Users/sdz/Desktop/micheal zhang/networks/musculusGroupGenes', header=None, sep='\t', index_col=None)
# f6 = pd.read_csv('/Users/sdz/Desktop/micheal zhang/networks/sapiensGroupGenes', header=None, sep='\t', index_col=None)
# f7 = pd.read_csv('/Users/sdz/Desktop/micheal zhang/networks/sapiensIntPathGenes', header=None, sep='\t', index_col=None)

f2[0]=f2[0].str.upper()
f2[1]=f2[1].str.upper()
f1=f1[~f1.index.duplicated(keep='first')]
f1=f1#.iloc[:,-101:]
f1.index=f1.index.str.upper()
d1=f1.fillna(0)#.loc[np.sum(f1>0,axis=1)>20]#.iloc[:20]
Eca=[]
print(f2)
print(f1)
for i in d1.columns:
    for j in d1.index:
        if pd.isna(j):
            print(j)
            continue
        if pd.isna(d1[i][j]):
            continue
        elif np.abs(d1[i][j])>0:
            Eca.append([i,j,np.exp(d1[i][j])])
print(pd.DataFrame(Eca).shape)
pd.DataFrame(Eca).to_csv(os.path.join(work_dr,'cell_context.txt'),index=False,header=False)

Ega=[]
context=d1.index
for j in f2.index:
    if f2.iloc[j][1] in context:
        Ega.append([f2.iloc[j][0],f2.iloc[j][1],1])
    elif f2.iloc[j][0] in context:
        Ega.append([f2.iloc[j][1],f2.iloc[j][0],1])
pd.DataFrame(Ega).to_csv(os.path.join(work_dr,'gene_context.txt'),index=False,header=False)
print(len(set([i[0] for i in Ega])))

#
# with open('cell-context.txt','w+') as out1:
#     for i in d1.columns:
#         for j in d1.index:
#             out1.writable([i,j,d1[i][j]])

# print('net genes',len(set(pd.concat([f4[1],f5[1],f6[1],f7[1]]).str.upper())))
# G = nx.Graph()

# f3[0]=f3[0].str.upper()
# f3[1]=f3[1].str.upper()
# f2[0]=f2[0].str.upper()
# f2[1]=f2[1].str.upper()
# f=pd.concat([f2[[0, 1]],f3[[0,1]]])
# print(f[~f.duplicated()].shape)
# pairs1 = np.array(f)
# print(pairs1.shape)
# for p in pairs1:
#     G.add_edge(p[0], p[1])
#
# # pairs2 = np.array(f4[[0, 1]])
# # for p in pairs2:
# #     G.add_edge(p[0], p[1],{'weight':1})
#
#
# set1 = set(G.nodes())
# set2 = set(f1.index.str.upper())
# con_genes = set1.intersection(set2)
# print(len(set(f4[1].str.upper()).intersection(set2)))
# print('net genes:',len(set1),'omics genes:', len(set2),'context genes:', len(con_genes),'edges:',G.number_of_edges())
#
# # cell_con=[]
# # for gene in f1.index:
# #     if gene not in con_genes:
# #         continue
# #     # print(f1.loc[gene])
# #     # print(f1.loc[gene]>0)
# #     nonzero = f1.columns[f1.loc[gene] > 500]
# #     for sam in nonzero:
# #         # print(f1.loc[gene][sam])
# #         cell_con.append([sam,gene,f1.loc[gene][sam]])
# # pd.DataFrame(cell_con).to_csv('/Users/sdz/Downloads/OPRL-master/cell_context.txt',sep='\t',header=False,index=False)
# #
# # gene_con=[]
# # for pair in G.edges():
# #     if pair[0] not in con_genes and pair[1] in con_genes:
# #         gene_con.append([pair[0],pair[1],G.edge[pair[0]][pair[1]]['weight']])
# # pd.DataFrame(gene_con).to_csv('/Users/sdz/Downloads/OPRL-master/gene_context.txt',sep='\t',header=False,index=False)
# G=nx.DiGraph()
# pairs1 = np.array(f2[[0, 1]])
# for p in pairs1:
#     if p[0] in con_genes:
#         G.add_edge(p[1], p[0], {'type': 'gene_context'})
#     else:
#         G.add_edge(p[0], p[1],{'type':'gene_context'})
#
#
# for gene in f1.index:
#     if gene not in con_genes:
#         continue
#     # print(f1.loc[gene])
#     # print(f1.loc[gene]>0)
#     nonzero=f1.columns[f1.loc[gene]>500]
#     for sam in nonzero:
#         # print(f1.loc[gene][sam])
#         G.add_edge(sam,gene,{'type':'cell_context'})
# net=[]
# print(G.number_of_nodes(),G.number_of_edges())
# for tup in G.edges():
#     net.append([tup[0],G.edge[tup[0]][tup[1]]['type'],tup[1]])
# # pd.DataFrame(net).to_csv('/Users/sdz/Downloads/net.tsv',sep='\t',header=False,index=False)
