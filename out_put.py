import torch
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
import os, json, time
from typing import Dict, Mapping, Optional
from collections import OrderedDict
from constants import (
    ENTITY_TO_EMBEDDING, ENTITY_TO_ID, EVAL_SUMMARY, FINAL_CONFIGURATION, LOSSES, OUTPUT_DIREC, RELATION_TO_EMBEDDING,
    RELATION_TO_ID, TRAINED_MODEL,
)
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from adjustText import adjust_text

def prepare_output(train, model):
    entity_label_to_id = train.vocab
    entity_label_to_embedding = {
        entity_label: model.entity_embeddings.weight.data[entity_label_to_id[entity_label]]
        for entity_label in entity_label_to_id
    }
    return _make_results(
        trained_model=model,
        entity_to_embedding=entity_label_to_embedding,
        entity_to_id=entity_label_to_id,
    )


def _make_results(trained_model,
                  entity_to_embedding: Mapping[str, np.ndarray],
                  entity_to_id) -> OrderedDict:
    results = OrderedDict()
    results[TRAINED_MODEL] = trained_model
    results[ENTITY_TO_EMBEDDING]: Mapping[str, np.ndarray] = entity_to_embedding
    results[ENTITY_TO_ID] = entity_to_id
    results[FINAL_CONFIGURATION] = trained_model.final_configuration
    return results


def save_results(results: OrderedDict, config: Dict,
                 output_directory: Optional[str] = None):
    if output_directory is None:
        output_directory = os.path.join(config[OUTPUT_DIREC], time.strftime("%Y-%m-%d_%H:%M:%S"))
    os.makedirs(output_directory, exist_ok=True)

    with open(os.path.join(output_directory, 'configuration.json'), 'w') as file:
        # In HPO model inital configuration is different from final configurations, thats why we differentiate
        json.dump(results[FINAL_CONFIGURATION], file, indent=2)

    # with open(os.path.join(output_directory, 'entities_to_embeddings.pkl'), 'wb') as file:
    #     pickle.dump(results[ENTITY_TO_EMBEDDING], file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output_directory, 'entities_to_embeddings.json'), 'w') as file:
        json.dump(
            {
                key: list(map(float, array))
                for key, array in results[ENTITY_TO_EMBEDDING].items()
            },
            file,
            indent=2,
            sort_keys=True
        )

    with open(os.path.join(output_directory, 'entity_to_id.json'), 'w') as file:
        json.dump(results[ENTITY_TO_ID], file, indent=2, sort_keys=True)

    # Save trained model
    torch.save(
        results[TRAINED_MODEL].state_dict(),
        os.path.join(output_directory, 'trained_model.pkl'),
    )


def scatterLegend(data, colors, x, y, labels: list):
    types = {key: [] for key in np.unique(colors)}
    for i in range(len(colors)):
        for j in np.unique(colors):
            if colors[i] == j:
                types[j].append(np.array(data[i]))
    gs = []
    for type in types:
        type_data = np.array(types[type])
        gs.append(plt.scatter(type_data[:, x], type_data[:, y]))
    plt.legend(handles=gs, labels=labels, loc='upper right')


def scatter_figure(output_directory,figname,pca, data, colors, x, y, labels: list):
    plt.figure()  # 新建一张图进行绘制
    scatterLegend(data, colors, x, y, labels)
    # for i input_data_guo range(result[:,0].size):
    #     plt.text(result[i,0],result[i,1],df.index[i])     #在每个点边上绘制数据名称
    x_label = 'PC1(%s%%)' % round((pca.explained_variance_ratio_[0] * 100.0), 2)  # x轴标签字符串
    y_label = 'PC2(%s%%)' % round((pca.explained_variance_ratio_[1] * 100.0), 2)  # y轴标签字符串
    plt.xlabel(x_label)  # 绘制x轴标签
    plt.ylabel(y_label)  # 绘制y轴标签
    plt.savefig(os.path.join(output_directory, figname))

def scatter_figure_2d(output_directory,figname,pca, x, y, colors, labels: list):
    color_pool=['r','b','g','y','purple','orange','pink','gray','black','chocolate','gold','navy','khaki','rosybrown','orangered','indigo','olive','lemonchiffon','lightcoral','lightpink','lightskyblue','forestgreen','darkred','darkorange']
    plt.figure()  # 新建一张图进行绘制
    # for i input_data_guo range(result[:,0].size):
    #     plt.text(result[i,0],result[i,1],df.index[i])     #在每个点边上绘制数据名称
    unique_colors = np.unique(colors)
    for i in range(len(unique_colors)):
        index = np.where(colors == unique_colors[i])
        plt.scatter(x[index], y[index], c=[color_pool[i]] * len(index), label=labels[i], s=30)
    x_label = 'PC1(%s%%)' % round((pca.explained_variance_ratio_[0] * 100.0), 2)  # x轴标签字符串
    y_label = 'PC2(%s%%)' % round((pca.explained_variance_ratio_[1] * 100.0), 2)  # y轴标签字符串
    plt.xlabel(x_label)  # 绘制x轴标签
    plt.ylabel(y_label)  # 绘制y轴标签
    plt.savefig(os.path.join(output_directory, figname))

def scatter_figure_3d(output_directory, figname, pca, X, Y, Z, colors, labels: list):
    color_pool=['r','b','g','y','purple','orange','pink','gray','black','chocolate','gold','navy','khaki','rosybrown','orangered','indigo','olive','lemonchiffon','lightcoral','lightpink','lightskyblue','forestgreen','darkred','darkorange']
    plt.figure(figsize=[10,8],)
    ax4 = plt.axes(projection='3d')
    colors = np.array(colors)
    # X = np.array(X)
    # Y = np.array(Y)
    # Z = np.array(Z)
    unique_colors=np.unique(colors)
    for i in range(len(unique_colors)):
        index = np.where(colors == unique_colors[i])
        ax4.scatter(X[index], Y[index], Z[index], c=[color_pool[i]]*len(index), label=labels[i],s=30)
    # ax4.scatter(X, Y, Z, alpha=0.3, c=colors)
    ax4.legend(loc=5,bbox_to_anchor=(1.2,1))
    x_label = 'PC1(%s%%)' % round((pca.explained_variance_ratio_[0] * 100.0), 2)  # x轴标签字符串
    y_label = 'PC2(%s%%)' % round((pca.explained_variance_ratio_[1] * 100.0), 2)  # y轴标签字符串
    z_label = 'PC3(%s%%)' % round((pca.explained_variance_ratio_[2] * 100.0), 2)  # y轴标签字符串
    ax4.set_xlabel(x_label)  # 绘制x轴标签
    ax4.set_ylabel(y_label)  # 绘制y轴标签
    ax4.set_zlabel(z_label)  # 绘制z轴标签
    plt.savefig(os.path.join(output_directory, figname),bbox_inches='tight')
    # plt.show()

def scatter_figure_3d_pathway(output_directory, figname, pca, data4scatter_df):
    # en2em=json.load(os.path.join(output_directory,'entities_to_embeddings.json'))
    # color_pool=['r','b','g','y','purple','orange','pink','gray','black','chocolate','gold','navy','khaki','rosybrown','orangered','indigo','olive','lemonchiffon','lightcoral','lightpink','lightskyblue','forestgreen','darkred','darkorange', 'gainsboro',
    # 'linen', 'bisque', 'darkseagreen', 'darkslateblue', 'darkviolet','royalblue','slategray']
    color_pool=['aqua','aquamarine','bisque','black','blue','blueviolet','brown','burlywood','cadetblue','chartreuse','chocolate','coral','cornflowerblue','crimson','cyan','darkblue','darkcyan','darkgoldenrod','darkgray','darkgreen','darkkhaki','darkmagenta','darkolivegreen','darkorange','darkorchid','darkred','darksalmon','darkseagreen','darkslateblue','darkslategray','darkturquoise','darkviolet','deeppink','deepskyblue','dimgray','dodgerblue','firebrick','forestgreen','fuchsia','gainsboro','ghostwhite','gold','goldenrod','gray','green','greenyellow','honeydew','hotpink','indianred','indigo','ivory','khaki','lavender','lavenderblush','lawngreen','lemonchiffon','lightblue','lightcoral','lightcyan','lightgoldenrodyellow','lightgreen','lightgray','lightpink','lightsalmon','lightseagreen','lightskyblue','lightslategray','lightsteelblue','lightyellow','lime','limegreen','linen','magenta','maroon','mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen','mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','mintcream','mistyrose','moccasin','navajowhite','navy','oldlace','olive','olivedrab','orange','orangered','orchid','palegoldenrod','palegreen','paleturquoise','palevioletred','papayawhip','peachpuff','peru','pink','plum','powderblue','purple','red','rosybrown','royalblue','saddlebrown','salmon','sandybrown','seagreen','seashell','sienna','silver','skyblue','slateblue','slategray','snow','springgreen','steelblue','tan','teal','thistle','tomato','turquoise','violet','wheat','white','whitesmoke','yellow','yellowgreen']
    plt.figure(figsize=[12,8],)
    ax4 = plt.axes(projection='3d')
    colors = data4scatter_df[0]
    # X = np.array(X)
    # Y = np.array(Y)
    # Z = np.array(Z)
    X = data4scatter_df[0]
    Y = data4scatter_df[1]
    Z = data4scatter_df[2]
    labels=list(data4scatter_df.index.values)
    # ax4.scatter(X, Y, Z, c=colors, label=labels,s=30)
    for i in range(len(X)):
        if 'cluster' in labels[i]:
            ax4.scatter(X[i], Y[i], Z[i], c=color_pool[i], label=labels[i],s=60,marker='^')
        else:
            ax4.scatter(X[i], Y[i], Z[i], c=color_pool[i], label=labels[i],s=30)
    # new_texts = [ax4.text(X[j], Y[j], Z[j], j, fontsize=9) for j in labels]
    # adjust_text(new_texts,
    #             only_move={'text': 'xy'},
    #             add_step_numbers=False,
    #             on_basemap=True)
    ax4.legend(loc=5, bbox_to_anchor=(1.8,.5))
    x_label = 'PC1(%s%%)' % round((pca.explained_variance_ratio_[0] * 100.0), 2)  # x轴标签字符串
    y_label = 'PC2(%s%%)' % round((pca.explained_variance_ratio_[1] * 100.0), 2)  # y轴标签字符串
    z_label = 'PC3(%s%%)' % round((pca.explained_variance_ratio_[2] * 100.0), 2)  # y轴标签字符串
    ax4.set_xlabel(x_label)  # 绘制x轴标签
    ax4.set_ylabel(y_label)  # 绘制y轴标签
    ax4.set_zlabel(z_label)  # 绘制z轴标签
    plt.savefig(os.path.join(output_directory, figname),bbox_inches='tight')
    # plt.show()

def scatter_figure_3d_gene(output_directory, figname, pca, data2scatter_df):
    # en2em=json.load(os.path.join(output_directory,'entities_to_embeddings.json'))
    color_pool=['aqua','aquamarine','bisque','black','blue','blueviolet','brown','burlywood','cadetblue','chartreuse','chocolate','coral','cornflowerblue','crimson','cyan','darkblue','darkcyan','darkgoldenrod','darkgray','darkgreen','darkkhaki','darkmagenta','darkolivegreen','darkorange','darkorchid','darkred','darksalmon','darkseagreen','darkslateblue','darkslategray','darkturquoise','darkviolet','deeppink','deepskyblue','dimgray','dodgerblue','firebrick','forestgreen','fuchsia','gainsboro','ghostwhite','gold','goldenrod','gray','green','greenyellow','honeydew','hotpink','indianred','indigo','ivory','khaki','lavender','lavenderblush','lawngreen','lemonchiffon','lightblue','lightcoral','lightcyan','lightgoldenrodyellow','lightgreen','lightgray','lightpink','lightsalmon','lightseagreen','lightskyblue','lightslategray','lightsteelblue','lightyellow','lime','limegreen','linen','magenta','maroon','mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen','mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','mintcream','mistyrose','moccasin','navajowhite','navy','oldlace','olive','olivedrab','orange','orangered','orchid','palegoldenrod','palegreen','paleturquoise','palevioletred','papayawhip','peachpuff','peru','pink','plum','powderblue','purple','red','rosybrown','royalblue','saddlebrown','salmon','sandybrown','seagreen','seashell','sienna','silver','skyblue','slateblue','slategray','snow','springgreen','steelblue','tan','teal','thistle','tomato','turquoise','violet','wheat','white','whitesmoke','yellow','yellowgreen']
    plt.figure(figsize=[10,8],)
    ax4 = plt.axes(projection='3d')
    colors = data2scatter_df[0]
    X = data2scatter_df[0]
    Y = data2scatter_df[1]
    Z = data2scatter_df[2]
    labels = data2scatter_df.index
    for i in range(len(X)):
        if 'cluster' in labels[i]:
            ax4.scatter(X[i], Y[i], Z[i], c=color_pool[i], label=labels[i],s=60,marker='^')
        else:
            ax4.scatter(X[i], Y[i], Z[i], c=color_pool[i], label=labels[i],s=30)
    # for j in labels:
    #     ax4.text(X[j]* 1.1, Y[j]* 1.1, Z[j] * 1.1, j, fontsize=9, color="r")  # 给散点加标签

    # new_texts = [ax4.text(X[j], Y[j], Z[j], j, fontsize=9) for j in labels]
    # adjust_text(new_texts,
    #             only_move={'text': 'xy'},
    #             add_step_numbers=False,
    #             on_basemap=True)

        # pathways=type_pathways.loc[np.where(type_pathways['cluster']==str(i+1))].sort_values('correlation',ascending=False).iloc[:3,:]['pathway'].values
    ax4.legend(loc=5, bbox_to_anchor=(1.4,.5))

    x_label = 'PC1(%s%%)' % round((pca.explained_variance_ratio_[0] * 100.0), 2)  # x轴标签字符串
    y_label = 'PC2(%s%%)' % round((pca.explained_variance_ratio_[1] * 100.0), 2)  # y轴标签字符串
    z_label = 'PC3(%s%%)' % round((pca.explained_variance_ratio_[2] * 100.0), 2)  # y轴标签字符串
    ax4.set_xlabel(x_label)  # 绘制x轴标签
    ax4.set_ylabel(y_label)  # 绘制y轴标签
    ax4.set_zlabel(z_label)  # 绘制z轴标签
    plt.savefig(os.path.join(output_directory, figname),bbox_inches='tight')
    # plt.show()

def calculate_average_distance(cluster1:np.ndarray,cluster2:np.ndarray):
    """:param cluster1: np.array,n1 samples*d dims
    :param cluster2: np.array,n2 samples*d dims
    """
    n1=cluster1.shape[0]
    n2=cluster2.shape[0]
    total=0
    for i in range(n1):
        for j in range(n2):
            distance=np.linalg.norm(cluster1[i]-cluster2[j])
            total+=distance
    return total/n1/n2


def WB_ratio(arrays:np.ndarray,labels:list):
    """:param arrays: ndarray,n samples*d dims
    :param labels: list,"""
    distance_inner=0
    distance_inter=0
    uniques=set(labels)
    c=len(uniques)
    clusters=[]
    for label in uniques:
        indices = [i for i, x in enumerate(labels) if x == label]
        X = arrays[indices, :]
        clusters.append(X)
    for i in range(c):
        cluster1=clusters[i]
        distance_inner+=calculate_average_distance(cluster1,cluster1)
        for j in range(i+1,c):
            cluster2=clusters[j]
            distance_inter+=calculate_average_distance(cluster1,cluster2)
    return distance_inner/c/(distance_inter/(c*(c-1)/2))

def find_significant_pathway_for_each_cell_type(model_dir,entity_embeddings:Dict,k):
    """:param cell_types:dict,like {type:[cells]}
    :param pathway_genes:dict,like {pathway:[genes]}
    :param entity_embeddings:dict,like{entity:embeddings}
    """
    embeddings={}
    df = pd.read_csv(os.path.join(model_dir,'result/ConsensusResult.csv'), sep=',')
    with open('pathway_genes.json', 'r') as f:
        pathway_genes = json.load(f)
    # print(df)
    cell_types = {}
    for i in df.index:
        line = df.iloc[i]
        cell = line['sample']
        type = line['k=%d'%k].astype('str')
        cell_types.setdefault(type, []).append(cell)
    type_pathways=[]
    for type in cell_types:
        # print(type)
        type_embeddings=[]
        cells=cell_types[type]
        for cell in cells:
            cell_embedding=entity_embeddings[cell].numpy()
            type_embeddings.append(cell_embedding)
        type_embedding=np.mean(type_embeddings,axis=0)
        embeddings[type]=type_embedding
        # embeddings[type]=type_embedding
        for pathway in pathway_genes:
            gene_embeddings=[]
            genes=pathway_genes[pathway]
            if len(genes)<=10:
                continue
            for gene in genes:
                try:
                    gene_embedding=entity_embeddings[gene].numpy()
                    gene_embeddings.append(gene_embedding)
                except:
                    pass
            pathway_embedding=np.mean(gene_embeddings,axis=0)
            embeddings[pathway]=pathway_embedding
            # data = pd.DataFrame([type_embedding,pathway_embedding])
            # x1=pd.Series(type_embedding)
            # x2=pd.Series(pathway_embedding)
            # print(x1.corr(x2,method='spearman'))
            # type_pathways[(type,pathway)]=ss.spearmanr(type_embedding,pathway_embedding)
            try:
                result=ss.spearmanr(type_embedding, pathway_embedding)
                type_pathways.append([type,pathway,result.correlation,result.pvalue])
                # print(type_embedding*pathway_embedding)
                # type_pathways.append([type,pathway,np.linalg.norm(type_embedding-pathway_embedding)])
            except:
                pass
    type_pathways=pd.DataFrame(type_pathways,columns=['cluster','pathway','correlation','p_value'])
    return type_pathways

def find_significant_genes_for_each_cell_type(model_dir,entity_embeddings:Dict,k):
    """:param cell_types:dict,like {type:[cells]}
    :param pathway_genes:dict,like {pathway:[genes]}
    :param entity_embeddings:dict,like{entity:embeddings}
    """
    df = pd.read_csv(os.path.join(model_dir,'result/ConsensusResult.csv'), sep=',')
    # print(df)
    cell_types = {}
    embeddings={}
    for i in df.index:
        line = df.iloc[i]
        cell = line['sample']
        type = line['k=%d'%k].astype('str')
        cell_types.setdefault(type, []).append(cell)
    type_genes=[]
    for type in cell_types:
        type_embeddings=[]
        cells=cell_types[type]
        for cell in cells:
            cell_embedding=entity_embeddings[cell].numpy()
            type_embeddings.append(cell_embedding)
        type_embedding=np.mean(type_embeddings,axis=0)
        embeddings[type]=type_embedding
        # embeddings[type]=type_embedding
        for gene in entity_embeddings:
            if gene in df['sample'].values:
                continue
            gene_embedding=entity_embeddings[gene].numpy()
            # type_genes.append([type,gene,np.linalg.norm(type_embedding-gene_embedding)])
            result=ss.spearmanr(type_embedding,gene_embedding)
            type_genes.append([type,gene,result.correlation,result.pvalue])
    type_genes=pd.DataFrame(type_genes,columns=['cluster','gene','correlation','p_value'])

    return type_genes

if __name__ == '__main__':
    # set configures
    input_dir = 'input_data_he_T'
    # cell_types={'ICMr_each_cell_type(cell_types,pathway_genes,entity_embeddings=results[ENTITY_TO_EMBEDDING])':['ICM_EPI_sc1','ICM_EPI_sc2','ICM_EPI_sc3','ICM_EPI_sc4','ICM_PE_sc1','ICM_PE_sc2','ICM_PE_sc3','ICM_PE_sc4','ICM_PE_sc5','ICM_PE_sc6','ICM_PE_sc7']}
    #     # pathway_genes={'Arachidonic acid metabolism':['PTGES2','CYP2J2','CYP2C19','PTGS2','CYP2C18','PTGS1','GGT1','LTC4S','AKR1C3','GPX2','GPX1'	]}
    #     # find_significant_pathway_fo