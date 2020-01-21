import pickle
import os
from annoy import AnnoyIndex
import numpy as np
import random
from matplotlib import pyplot as plt
import collections


def build_graph(cluster_embedding_graph, save_dir, by_centroid = True):
	t = AnnoyIndex(200, metric='angular')
	index = 0
	index_to_centroid = {}

	for word in cluster_embedding_graph:
		num_c = len(cluster_embedding_graph[word])
		if by_centroid:
			for c in range(num_c):
				t.add_item(index, cluster_embedding_graph[word][c]['centroid'])
				index_to_centroid[index] = {}
				index_to_centroid[index]['word'] = word
				index_to_centroid[index]['contexts'] = cluster_embedding_graph[word][c]['contexts']
				index += 1
		else:
			for c in range(num_c):
				embeddings = cluster_embedding_graph[word][c]['embeddings']
				for e in range(len(embeddings)):
					t.add_item(index, embeddings[e])
					index_to_centroid[index] = {}
					index_to_centroid[index]['word'] = word
					index_to_centroid[index]['contexts'] = cluster_embedding_graph[word][c]['contexts'][e]
					index += 1
	t.build(50)
	tree_name = os.path.join(save_dir, "clustered_annoy_graph.ann")
	t.save(tree_name)
	with open(os.path.join(save_dir, 'index_to_word.pkl'), 'wb') as f:
		pickle.dump(index_to_centroid, f, pickle.HIGHEST_PROTOCOL)


def analyze_graph(save_dir):
	t = AnnoyIndex(200)
	t.load(os.path.join(save_dir, "clustered_annoy_graph.ann"))
	f = open(os.path.join(save_dir, 'index_to_word.pkl'), 'rb')
	index_to_word = pickle.load(f)
	run_indices = []
	for index in index_to_word:
		if index_to_word[index]['word'] == 'run':
			run_indices.append(index)
	print(run_indices)
	for y in run_indices:
		nns = t.get_nns_by_item(y, 20)
		word = index_to_word[y]['word']
		context = index_to_word[y]['contexts']
		nns_words = [index_to_word[x]['word'] for x in nns]
		print(word, context)
		print(nns_words)