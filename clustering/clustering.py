# -*- coding: utf-8 -*-
import math
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from heapq import heappush, heappop, nsmallest
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def load_embeddings(words_to_plot=[]):
	words_to_embeddings={x:{'embeddings':[],'words':[]} for x in words_to_plot}
	run=0
	while True:
		if run %1000 == 0:
			print ("Run: "+str(run))
		try:
			input_path = os.path.join(base_dir, file_dir + "input_"+str(run)+"_A.npy")
			input_file = np.load(input_path, encoding='bytes')

			emb_path = os.path.join(base_dir, file_dir + "emb_"+str(run)+"_A.npy")
			emb_file = np.load(emb_path, encoding='bytes')

			for x in range(len(input_file)):
				curWords = [inv_vocab_dict[word_idx]['word'] for word_idx in input_file[x][0]]
				centerWord = curWords[2]
				curEmbedding = emb_file[x]
				if centerWord in words_to_plot:
					words_to_embeddings[centerWord]['embeddings'].append(curEmbedding)
					words_to_embeddings[centerWord]['words'].append(curWords)
		except FileNotFoundError:
			break
		run+=1
	return words_to_embeddings

def load_oed_embeddings():
	semcor_base_dir = "/usr/xtmp/kpg12/data/word_sense_disambigation_corpora/semcor_in_original_format"
	base_dir = "/usr/xtmp/ays7/wikipedia_tagged_25000"
	kenny_dir = "/usr/xtmp/kpg12/data/wikipedia_tagged_25000"
	model_dir = "0217_matching_word_q_pairing"
	saved_model_dir = os.path.join(base_dir, model_dir, "trained_model")
	save_dir = os.path.join(kenny_dir, model_dir, "google_sense_embeddings")
	src_dir2 = os.path.join(kenny_dir, "google_test")

	word_sense_embeddings_map = {}
	sense_embeddings_map = np.load(os.path.join(save_dir, "sense_embeddings_map.npy")).item()
	sense_context_map = np.load(os.path.join(save_dir, "sense_context_map.npy")).item()
	words = 0
	for word_sense, embeddings in sense_embeddings_map.items():
		sense_contexts = sense_context_map[word_sense]
		for i in range(len(embeddings)):
			context = sense_contexts[i]
			center_word = context[2]
			if (center_word not in word_sense_embeddings_map):
				word_sense_embeddings_map[center_word] = {}
			if (word_sense not in word_sense_embeddings_map[center_word]):
				word_sense_embeddings_map[center_word][word_sense] = []
			# word_sense_embeddings_map[center_word][word_sense].append(embeddings[i])
			word_sense_embeddings_map[center_word][word_sense].append([embeddings[i], context])

	return word_sense_embeddings_map

def cluster_dbscan_binary(embs,contexts):
	best_db=None
	max_clusters=-1
	left=0
	right=5
	cur=.5
	vals=np.linspace(.01,1,100)
	best_runs=[]
	prev_clusters=1
	prev_eps=0
	while not math.isclose(left, right, rel_tol=1e-4):
		db = DBSCAN(eps=cur).fit(embs)
		labels = list(db.labels_)
		print(len(labels))
		print(os.getcwd())
		n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

		if n_clusters_ > max_clusters:
			max_clusters= n_clusters_
			pickle.dump( db, open( os.getcwd()+"/data/best_db.p", "wb" ) )
			# print(n_clusters_)

		#Eps is too big
		if n_clusters_ == 1:
			right=cur
			cur=(cur+left)/2.0

		#Eps is too small
		if n_clusters_ == 0:
			left=cur
			cur = (cur+right)/2.0

		if n_clusters_ > prev_clusters:
			if cur <= prev_eps:
				right=cur
				cur= (left+cur)/2.0
			if cur > prev_eps:
				left=cur
				cur= (right+cur)/2.0

		if n_clusters_ < prev_clusters:
			if cur > prev_eps:
				right=cur
				cur= (left+cur)/2.0
			if cur <= prev_eps:
				left=cur
				cur= (right+cur)/2.0


		if cur==prev_eps:
			break
		prev_clusters=n_clusters_
		prev_eps = cur

	db = pickle.load( open( os.getcwd()+"/data/best_db.p", "rb" ) )
	return db

def cluster_dbscan(embs,contexts):
	best_db=None
	max_clusters=-1
	low=0.1
	vals=np.linspace(.01,1,100)
	best_runs=[]
	for x in vals:
		db = DBSCAN(eps=x).fit(embs)
		labels = list(db.labels_)
		n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
		if n_clusters_ > max_clusters:
			max_clusters= n_clusters_
			pickle.dump(db, open( "best_db.p", "wb" ) )
			# print(n_clusters_)
	try:
		db = pickle.load( open( "best_db.p", "rb" ) )
		return db
	except FileNotFoundError:
		return None

def cluster_k_means(embs,contexts,n_clusters):
	kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embs)
	return kmeans

"""
	Returns predicted labels from cluster_obj
"""
def get_labels(cluster_obj):
	labels = list(cluster_obj.labels_)
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

	# print("created " + str(n_clusters_) +" clusters")
	# print("percentage of points",sum([1 if x>-1 else 0 for x in labels])/float(len(labels)))
	return labels

"""
	Returns clusters formatted for constructing annoy graph
"""
def get_annoy(cluster_obj,contexts,embs,verbose=True):
	labels = list(cluster_obj.labels_)
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	clustersToContext,clustersToEmbs = get_cluster_maps(n_clusters_,labels,contexts,embs)
	
	if verbose:
		for x in clustersToContext:
			for y in x:
				print(y)
			print()
			
		print([len(clustersToContext[x]) for x in range(len(clustersToContext))])

	output=annoy_format(clustersToContext,clustersToEmbs,n_clusters_)
	return output

"""
Returns x: 
		   x[0] = map from clusters to contexts
		   x[1] = map from clusters to embeddings
"""
def get_cluster_maps(n_clusters_,labels,contexts,embs):
	assert (
	    len(labels) == len(contexts) == len(embs)
	), "\nLengths (Should be equal): \n Labels: {},\n Contexts: {},\n Embeddings: {}".format(
	    len(labels), len(contexts), len(embs))

	rng = range(-1,n_clusters_+1) if -1 in labels else range(n_clusters_)
	clustersToContext = [[] for x in rng]
	clustersToEmbs = [[] for x in rng]

	notClusteredContexts = []
	notClusteredEmbs = []
	for x in rng:
		for y in range(len(labels)):
			if labels[y]==x:
				clustersToContext[x].append(contexts[y])
				clustersToEmbs[x].append(embs[y])

	return clustersToContext,clustersToEmbs


"""
	Returns output of clustering in format for creating annoy graph
"""
def annoy_format(clustersToContext,clustersToEmbs,n_clusters_):
	centroid_vals = [np.array(x).mean(axis=0) for x in clustersToEmbs]
	centroids={x:centroid_vals[x] for x in range(len(centroid_vals))}

	n_nearest=10

	output={n:{} for n in range(-1,n_clusters_)}
	for n in range(n_clusters_):
		vals=get_nearest_n(centroids[n],clustersToEmbs[n],n_nearest)
		inds = (vals.index)
		nearest_embs = vals.values

		nearest_contexts = [clustersToContext[n][x] for x in inds]
		output[n]['embeddings'] = nearest_embs
		output[n]['contexts'] = nearest_contexts
		output[n]['centroid'] = centroids[n]

	return output

"""
	Returns nearest n points to the centroid of a given list of points
"""
def get_nearest_n(centroid,cluster_points,n):
	if n>= len(cluster_points):
		return pd.DataFrame(cluster_points)

	points=pd.DataFrame(cluster_points)
	points['distance']=((points - centroid)**2).sum(axis=1) ** 0.5
	points.sort_values(by=['distance'],inplace = True)
	points=points.drop(columns=['distance'])
	return points.head(n)



if __name__ == "__main__":
	try:
		os.chdir('/usr/xtmp/jas198')
		#This is just codenames words
		f=open('codenames_embs.pkl', 'rb')
		embs=pickle.load(f)
		print("loaded pickle")
	except:
		embs=load_embeddings()
		print("regenerating pickle")

	#This is the word to cluster on --must be in the codenames word list
	words = ['cook']
	for word in words:
		print(len(embs[word]['words']))


		print("binary time")
		start = time.time()
		out = cluster_dbscan_binary(embs[word]['embeddings'],embs[word]['words'])
		# for x in range(10):
		# 	out = cluster_dbscan_binary(out[-1]['embeddings'],out[-1]['contexts'])
		end = time.time()
		print(end - start)