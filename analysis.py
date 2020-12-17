'''
consensus cluster for OPRL embedded vectors
pca representation
pathway finding
gene finding and visualization
'''
from out_put import *
from train import *
from sklearn import decomposition as skldec  # 用于主成分分析降维的包
import seaborn as sns
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score

# set configures
dirs = ['2020-11-28_17:24:14']

k = 6
for model_dir in dirs:
    print(model_dir)
    config = json.load(open(os.path.join(model_dir, 'configuration.json'), 'r'))
    input_dir=config['input_dir']


    embeddings = json.load(open(os.path.join(model_dir, 'entities_to_embeddings.json')))
    df = pd.DataFrame(embeddings)
    print(input_dir)
    cells = pd.read_csv(os.path.join(input_dir,'cells.csv'), usecols=[0], header=None)

    m = df[cells[0]]
    m.to_csv(os.path.join(model_dir, 'embedding_matrix.csv'))
    prior = config['prior_weight']
    train = TrainOPRL(config)
    model = train.model
    path_to_model = os.path.join(model_dir, 'trained_model.pkl')
    state_dict = torch.load(path_to_model)
    model.load_state_dict(state_dict)
    results = prepare_output(train, model)
    # prepare to evaluate

    cell_names = pd.DataFrame(train.Eca)[0].drop_duplicates().values
    cell_indices = [train.vocab[i] for i in cell_names]
    truth_cluster = pd.read_csv(os.path.join(input_dir, 'cells.csv'), header=None, index_col=0)
    colors = [truth_cluster.loc[i][1] for i in cell_names]
    labels = list(truth_cluster[2].unique())
    dt = model.entity_embeddings.weight.data
    df = pd.DataFrame(dt, dtype='float').iloc[cell_indices]

    # consensus survival
    os.chdir(os.path.join(os.getcwd(), model_dir))
    os.system('Rscript /Users/sdz/PycharmProjects/OPRL7/consensus_survival_our.R embedding_matrix.csv\
     /Users/sdz/PycharmProjects/OPRL7/input_cptac_6_na_exp/survival_hcc.csv 15 hc pearson %d'%k)
    os.chdir('/Users/sdz/PycharmProjects/OPRL7')


    # # 根据两个最大的主成分进行绘图 PCA
    pca = skldec.PCA(n_components=0.95)  # 选择方差95%的占比
    pca.fit(df)  # 主城分析时每一行是一个输入数据
    result = pca.transform(df)  # 计算结果
    X = result[:, 0]
    Y = result[:, 1]
    Z = result[:, 2]
    scatter_figure_3d(model_dir, 'result/fig_pca.pdf', pca, X, Y, Z, colors, labels)
    #
    #
    #
    # # #
    # # 对聚类之后的 聚类 结果进行PCA可视化
    consensus_results = pd.read_csv(os.path.join(model_dir, 'result/ConsensusResult.csv'), index_col=0)
    new_cell_names = [i.replace(':', '.').replace('-', '.').replace(' ', '.') for i in cell_names]
    new_colors = list(consensus_results.loc[new_cell_names]['k=%d' % k].values)
    new_labels = ['Prior_based_cluster%d' % (i + 1) for i in range(k)]
    scatter_figure_3d(model_dir, 'result/fig_pca_cluster.pdf', pca, X, Y, Z, new_colors, new_labels)


    # # 计算聚类准确率
    # def f(x):
    #     if x==1:
    #         return 1
    #     if x==2:
    #         return 0
    #     if x==3:
    #         return 2
    # y_pred = list(map(f,new_colors))
    # y_true = colors
    # with open(os.path.join(model_dir,'result/performance.txt'),'w+') as file:
    #     file.write('accuracy: '+str(accuracy_score(y_true, y_pred)))
    #     file.write('\nprecision: '+str(precision_score(y_true, y_pred, average='macro')))
    #     file.write('\nrecall: '+str(recall_score(y_true, y_pred, average='macro')))
    #     file.write('\nf1: '+str(f1_score(y_true, y_pred, average='macro')))
    # print('accuracy: ', accuracy_score(y_true, y_pred))
    # print('precision: ', precision_score(y_true, y_pred, average='macro'))
    # print('recall: ', recall_score(y_true, y_pred, average='macro'))
    # print('f1: ', f1_score(y_true, y_pred, average='macro'))

    # # #
    # # 准备通路、基因发现
    clusters = pd.read_csv(os.path.join(model_dir, 'result/ConsensusResult.csv'), sep=',')
    with open('pathway_genes.json', 'r') as f:
        pathway_genes = json.load(f)
    entity_embeddings = pd.DataFrame(results[ENTITY_TO_EMBEDDING]).T
    pca = skldec.PCA(n_components=3)  # 选择方差95%的占比
    pca.fit(entity_embeddings)  # 主成分分析时每一行是一个输入数据
    result = pca.transform(entity_embeddings)  # 计算结果
    index = entity_embeddings.index
    result_df = pd.DataFrame(result, index=index)

    # display pathway
    pathway_name='p53 signaling pathway'
    genes=pathway_genes[pathway_name]
    genes2scatter={}
    for gene in genes:
        if gene in result_df.index:
            genes2scatter[gene]=result_df.loc[gene]
    genes2scatter_df=pd.DataFrame(genes2scatter).T
    scatter_figure_3d_gene(model_dir,'%s.pdf'%pathway_name,pca,genes2scatter_df)
    #
    # # # WB-ratio
    # array = m.T.values
    # colors4wb = []
    # for i in m.columns:
    #     colors4wb.append(clusters['k=%d' % k][np.where(clusters['sample']==i)[0]].values[0])
    # wb_ratio = WB_ratio(array, colors4wb)
    # with open(os.path.join(model_dir, 'result/wb_ratio.txt'), 'w+') as f1:
    #     json.dump(wb_ratio, f1)
    # #
    # # # 通路发现
    type_pathways = find_significant_pathway_for_each_cell_type(model_dir,
                                                                entity_embeddings=results[ENTITY_TO_EMBEDDING], k=k)
    type_pathways.to_csv(os.path.join(model_dir, 'result/pathway_results.csv'))
    type_pathway = []
    data4scatter = {}
    result_emb={}
    labels4wb=[]
    for i in range(k):
        pathways = type_pathways.loc[np.where(type_pathways['cluster'] == str(i + 1))].sort_values('correlation',
                                                                                                   ascending=False).iloc[
                   :3, :]['pathway'].values
        cells = clusters['sample'][np.where(clusters['k=%d' % k] == i + 1)[0]]
        cell_pca = result_df.loc[cells].mean()
        cell_emb = entity_embeddings.loc[cells].mean()
        data4scatter['Prior_based_cluster' + str(i + 1)] = cell_pca
        result_emb['Prior_based_cluster' + str(i + 1)] = cell_emb
        labels4wb.append(i)
        genes = []

        for pathway in pathways:
            genes = pathway_genes[pathway]
            genes = list(set(genes).intersection(set(index)))
            genes_pca = result_df.loc[genes].mean()
            genes_emb = entity_embeddings.loc[genes].mean()
            data4scatter[pathway] = genes_pca
            if pathway not in result_emb :
                labels4wb.append(i)
                result_emb[pathway] = genes_emb

    data4scatter_df = pd.DataFrame(data4scatter).T
    result_emb_df = pd.DataFrame(result_emb).T
    wb_ratio_path=WB_ratio(np.array(result_emb_df.values),labels4wb)
    with open(os.path.join(model_dir,'result/wb_ratio_pathways.txt'),'w+') as f1:
        json.dump(wb_ratio_path,f1)

    # #
    result_emb_df.to_csv(os.path.join(model_dir,'result/pathway_embeddings.csv'))
    data4scatter_df.to_csv(os.path.join(model_dir, 'result/pathway_pca.csv'))
    scatter_figure_3d_pathway(model_dir, 'result/fig_pca_pathways.pdf', pca, data4scatter_df)
    #
    # # #基因发现
    type_genes = find_significant_genes_for_each_cell_type(model_dir,
                                                           entity_embeddings=results[ENTITY_TO_EMBEDDING], k=k)
    type_genes.to_csv(os.path.join(model_dir, 'result/gene_results.csv'))
    type_gene = []
    data2scatter = {}
    result_emb_gene={}
    for i in range(k):
        genes = type_genes.loc[np.where(type_genes['cluster'] == str(i + 1))].sort_values('correlation',
                                                                                          ascending=False).iloc[
                :3, :]['gene'].values
        cells = clusters['sample'][np.where(clusters['k=%d' % k] == i + 1)[0]]
        cell_pca = result_df.loc[cells].mean()
        cell_emb = entity_embeddings.loc[cells].mean()
        data2scatter['Prior_based_cluster' + str(i + 1)] = cell_pca
        result_emb_gene['Prior_based_cluster' + str(i + 1)] = cell_emb
        # cell_pca = result_df.loc[cells]
        # for x in cell_pca.index:
        #     data2scatter[x]=cell_pca.loc[x]
        for gene in genes:
            gene_pca = result_df.loc[gene]
            gene_emb=entity_embeddings.loc[gene]
            data2scatter[gene] = gene_pca
            result_emb_gene[gene] = gene_emb
    data2scatter_df = pd.DataFrame(data2scatter).T
    result_emb_gene_df = pd.DataFrame(result_emb_gene).T
    data2scatter_df.to_csv(os.path.join(model_dir, 'result/gene_pca.csv'))
    result_emb_gene_df.to_csv(os.path.join(model_dir, 'result/gene_embeddings.csv'))
    scatter_figure_3d_gene(model_dir, 'result/fig_pca_gene.pdf', pca, data2scatter_df)

    # if input_dir=='input_cptac':
    # hotmap gene
    f1 = pd.read_csv(os.path.join(input_dir,'profiles.csv'), sep=',', index_col=0)
    # f1 = pd.read_csv('/Users/sdz/Desktop/桌面/HCC/liver_cancer_iBAQ_Intensity_2019_nature_normalized的副本.csv', sep=',', index_col=0)
    # f1=np.log(f1)

    f1 = f1[~f1.index.duplicated(keep='first')]
    f1 = f1  # .iloc[:,-101:]
    f1.index = f1.index.str.upper()
    d1 = f1.fillna(0)
    consensus_results = pd.read_csv(os.path.join(model_dir, 'result/ConsensusResult.csv'), sep=',')
    # print(df)
    cell_types = {}
    embeddings = {}
    cluster_ex = pd.DataFrame()
    for i in consensus_results.index:
        line = consensus_results.iloc[i]
        cell = line['sample']
        type = line['k=%d' % k].astype('str')
        cell_types.setdefault(type, []).append(cell)

    samples2hotmap = []
    genes2hotmap = []
    for i in range(k):
        genes = type_genes.loc[np.where(type_genes['cluster'] == str(i + 1))].sort_values('correlation',
                                                                                          ascending=False).iloc[
                :3, :]['gene'].values
        for gene in genes:
            if gene in d1.index:
                genes2hotmap.append(gene)
    cols = []
    for type in cell_types:
        type_ex = f1[cell_types[type]].mean(axis=1)
        cluster_ex = pd.concat([cluster_ex, type_ex], axis=1)
        cols.append(eval(type))
    cluster_ex.columns = cols
    cluster_ex.sort_index(axis=1, inplace=True)
    # plt.figure(figsize=(10,8))
    f, ax = plt.subplots(figsize=(9, 6))
    genes_heat_data=cluster_ex.loc[genes2hotmap].T
    genes_heat_data.to_csv(os.path.join(model_dir,'result/hotmap_gene.csv'))
    ax = sns.heatmap(genes_heat_data.fillna(0),xticklabels = 1,cbar = True,cmap='RdBu_r')
    cbar = ax.collections[0].colorbar
    cbar.set_label('Mean SD')
    ax.set_xlabel('Proteins')
    ax.set_ylabel('Prior-based clusters')
    fig = ax.get_figure()
    fig.savefig(os.path.join(model_dir, 'result/fig_hotmap_gene.pdf'), bbox_inches='tight')

    # # hotmap pathway
    # f1 = pd.read_csv(os.path.join(input_dir,'profiles.csv'), sep=',', index_col=0) ###########注意调整
    # f1 = f1[~f1.index.duplicated(keep='first')]
    # f1 = f1  # .iloc[:,-101:]
    # f1.index = f1.index.str.upper()
    # d1 = f1.fillna(0)
    # consensus_results = pd.read_csv(os.path.join(model_dir, 'result/ConsensusResult.csv'), sep=',')
    # # print(df)
    # cell_types = {}
    # embeddings = {}
    # cluster_ex = pd.DataFrame()
    # for i in consensus_results.index:
    #     line = consensus_results.iloc[i]
    #     cell = line['sample']
    #     type = line['k=%d' % k].astype('str')
    #     cell_types.setdefault(type, []).append(cell)
    # samples2hotmap = []
    # pathways2hotmap = []
    # pathways_samples=pd.DataFrame()
    #
    # pathways_samples_index =[]
    # for i in range(k):
    #     pathways = type_pathways.loc[np.where(type_pathways['cluster'] == str(i + 1))].sort_values('correlation',
    #                                                                                                ascending=False).iloc[
    #                :3, :]['pathway'].values
    #
    #     for pathway in pathways:
    #         genes = pathway_genes[pathway]
    #         genes = list(set(genes).intersection(set(d1.index)))
    #         genes_ex = d1.loc[genes].mean()
    #         pathways_samples=pd.concat([pathways_samples,genes_ex],axis=1)
    #         pathways_samples_index.append(pathway)
    # pathways_samples=pathways_samples.T
    # pathways_samples.index=pathways_samples_index
    # pathways_samples.columns = d1.columns
    # cols = []
    # for type in cell_types:
    #     type_ex = pathways_samples[cell_types[type]].mean(axis=1).dropna().drop_duplicates()
    #     cluster_ex = pd.concat([cluster_ex, type_ex], axis=1)
    #     cols.append(eval(type))
    # cluster_ex.columns = cols
    # cluster_ex.sort_index(axis=1, inplace=True)
    # # plt.figure(figsize=(10,8))
    # f, ax = plt.subplots(figsize=(9, 6))
    # cluster_ex.T.to_csv(os.path.join(model_dir,'result/hotmap_pathway.csv'))
    # ax = sns.heatmap(cluster_ex.T,xticklabels = 1,cbar = True,cmap='RdBu_r')
    # cbar = ax.collections[0].colorbar
    # cbar.set_label('Mean SD')
    # ax.set_xlabel('Pathways')
    # ax.set_ylabel('Prior-based clusters')
    # fig = ax.get_figure()
    # fig.savefig(os.path.join(model_dir, 'result/fig_hotmap_pathway.pdf'), bbox_inches='tight')
