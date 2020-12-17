from out_put import *
from train import TrainOPRL
import pandas as pd
import seaborn as sns  # 用于绘制热图的工具包
from scipy.cluster import hierarchy  # 用于进行层次聚类，话层次聚类图的工具包
from scipy import cluster
# from sklearn import decomposition as skldec  # 用于主成分分析降维的包

# set configures
input_dir = 'input_data'
output_dir = ''
for j in [0]:
    for i in [15]:
        config = {'input_dir': input_dir, 'margin_loss': -0.62, 'embedding_dim':  i , 'lp_norm': 2,
                  'T': 2000, 'K': 5, 'learning_rate': .1, 'output_direc': output_dir,'preferred_device':'cpu','prior_weight':j}

        # train model
        train = TrainOPRL(config)
        model = TrainOPRL.run(train)
        results = prepare_output(train, model)
        output_directory = os.path.join(config[OUTPUT_DIREC], time.strftime("%Y-%m-%d_%H:%M:%S"))
        save_results(results, config, output_directory)

        # # prepare to evaluate
        # cell_names = pd.DataFrame(train.Eca)[0].drop_duplicates()
        # cell_indices = [train.vocab[i] for i in cell_names]
        # truth_cluster = pd.read_csv(os.path.join(input_dir, 'cells.csv'), header=None, index_col=0)
        # colors = [truth_cluster.loc[i][1] for i in cell_names]
        # labels = list(truth_cluster[2].unique())
        # dt = model.entity_embeddings.weight.data
        # df = pd.DataFrame(dt, dtype='float').iloc[cell_indices]

        # #hierarchy
        # Z = hierarchy.linkage(np.array(df), method='ward', metric='euclidean')
        # hierarchy.dendrogram(Z, labels=np.array([i[0] for i in train.Eca]))
        # plt.savefig(os.path.join(output_directory, 'fig_hierarchy.png'))
        # label = cluster.hierarchy.cut_tree(Z, n_clusters=[len(labels)])
        # label = label.reshape(label.size, )
        # l = []
        # for i, j in zip(cell_names, label):
        #     l.append([i, j])
        # l = pd.DataFrame(l)
        # l.to_csv(os.path.join(output_directory, 'cluster.csv'),header=False,index=False)

        # # 根据两个最大的主成分进行绘图 PCA
        # pca = skldec.PCA(n_components=0.95)  # 选择方差95%的占比
        # pca.fit(df)  # 主城分析时每一行是一个输入数据
        # result = pca.transform(df)  # 计算结果
        # scatter_figure(output_directory,'fig_pca.png', pca,result, colors, 0, 1, labels)
        #
        # # WB-ratio
        # array=result[:,0:2]
        # wb_ratio=WB_ratio(array,colors)
        # with open(os.path.join(output_directory,'wb_ratio.txt'),'w+') as f1:
        #     json.dump(wb_ratio,f1)

        # # hotmap
        # sns.clustermap(df, method='ward', metric='euclidean')
        # plt.savefig(os.path.join(output_directory, 'fig_hotmap.png'))


        # # find pathway
        # with open('cell_types.json','r') as f1:
        #     cell_types=json.load(f1)
        # with open('pathway_genes.json','r') as f:
        #     pathway_genes=json.load(f)
        #
        # type_pathways=find_significant_pathway_for_each_cell_type(cell_types,pathway_genes,entity_embeddings=results[ENTITY_TO_EMBEDDING])
        # type_pathways.to_csv(os.path.join(output_directory,'pathway_results.csv'))
        #
        #
        # # embedding matrix
        # embeddings=json.load(open(os.path.join(output_directory,'entities_to_embeddings.json')))
        # df=pd.DataFrame(embeddings)
        #
        # cells=pd.read_csv('input_data_he_T/cells.csv',usecols=[0],header=None)
        #
        # m=df[cells[0]]
        # m.to_csv(os.path.join(output_directory,'embedding_matrix.csv'))

        # # consensus survival
        # os.chdir(os.path.join(os.getcwd(),output_directory))
        # os.system('Rscript ../consensus_survival.R embedding_matrix.csv ../%s/survival1.csv 10 hc pearson' % input_dir)
        #
        #
