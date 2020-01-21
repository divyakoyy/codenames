from sklearn import metrics
from clustering import *
import numpy as np
from clustering_build_graph import build_graph, analyze_graph

def score_clustering_method(embeddings,contexts,true_labels,clusteringMethod,n_clusters=-1):
	# as many labels as we have data points
	assert len(embeddings) == len(contexts) == len(true_labels)

	if n_clusters == -1:
		# Unsupervised
		cluster_obj = clusteringMethod(embeddings, contexts)
	else:
		# Supervised
		cluster_obj = clusteringMethod(embeddings, contexts, n_clusters)

	# print(len(embeddings),len(contexts),len(true_labels),len(cluster_obj.labels_))
	predicted_labels = get_labels(cluster_obj)    


	AMI_score = metrics.adjusted_mutual_info_score(true_labels, predicted_labels)
	homogeneity_score = metrics.homogeneity_score(true_labels, predicted_labels)
	fowlkes_mallows_score = metrics.fowlkes_mallows_score(true_labels, predicted_labels)
	adjusted_rand_score = metrics.cluster.adjusted_rand_score(true_labels, predicted_labels)

	print("Scores: \n AMI:", AMI_score, '\n',
	      "Homogeneity:", homogeneity_score, '\n',
	      "Fowlkes_mallows_score:", fowlkes_mallows_score)

def get_annoy_format(embeddings,contexts,true_labels,clusteringMethod,n_clusters=-1):
	# as many labels as we have data points
	assert len(embeddings) == len(contexts) == len(true_labels)

	if n_clusters == -1:
		# Unsupervised
		cluster_obj = clusteringMethod(embeddings, contexts)
	else:
		# Supervised
		cluster_obj = clusteringMethod(embeddings, contexts, n_clusters)

	annoy_format = get_annoy(cluster_obj, contexts, embeddings, verbose=False)

	del annoy_format[-1]
	return annoy_format

def build_annoy_graph(word_embeddings, save_dir, by_centroid = False):
	words_to_annoy = {}
	for word in word_embeddings.keys():
		label_num = 0
		embs = []
		contexts = []
		labels = []
		for sense in word_embeddings[word]:
			print(sense)
			for x in word_embeddings[word][sense]:
				emb, context = x[0], x[1]
				embs.append(emb)
				contexts.append(context)
				labels.append(label_num)

			label_num += 1
		annoy = get_annoy_format(embs, contexts, labels, cluster_k_means,
		                                n_clusters=len(np.unique(np.array(labels))))
		words_to_annoy[word] = annoy
	build_graph(words_to_annoy, save_dir, by_centroid=by_centroid)


if __name__ == "__main__":
	word_embeddings = load_oed_embeddings()

	# for annoy processing
	save_dir = '/usr/xtmp/kpg12/data/annoy_graphs/'
	build_annoy_graph(word_embeddings, save_dir, False)
	analyze_graph(save_dir)

	# analyze clustering of specific words
	# words_to_cluster = ['bank', 'run']
	# label_num = 0
	# for word in words_to_cluster:
	# 	embs = []
	# 	contexts = []
	# 	labels = []
	# 	for sense in word_embeddings[word]:
	# 		for x in word_embeddings[word][sense]:
	# 			emb, context = x[0], x[1]
	# 			embs.append(emb)
	# 			contexts.append(context)
	# 			labels.append(label_num)
	# 		label_num += 1
	# 	print(label_num, "labelNum")
	# 	score_clustering_method(embs, contexts, labels, cluster_k_means, n_clusters=len(np.unique(np.array(labels))))