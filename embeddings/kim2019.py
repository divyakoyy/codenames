import itertools
import numpy as np
import utils as utils

import gensim.models.keyedvectors as word2vec
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from scipy.spatial.distance import cosine as cos_dist


class Kim2019(object):
	def __init__(
		self,
		configuration=None,
		word_to_dict2vec_embeddings=None,
		w2v_file_path='data/GoogleNews-vectors-negative300.bin'
	):
		super().__init__()
		self.word_to_dict2vec_embeddings = word_to_dict2vec_embeddings
		self.configuration = configuration
		# self.word_vectors = word2vec.KeyedVectors.load_word2vec_format(w2v_file_path, binary=True, unicode_errors='ignore')
		self.word_vectors = word2vec.KeyedVectors.load_word2vec_format('data/glove-wiki-gigaword-300.txt.gz')
		self.wordnet_lemmatizer = WordNetLemmatizer()
		self.lancaster_stemmer = LancasterStemmer()
		self.cm_cluelist = []  # candidate clues

		with open('data/cm_wordlist.txt') as infile:
			for line in infile:
				self.cm_cluelist.append(line.rstrip())
		self.word_dists = {}

	def get_weighted_nn(self, word):
		nn_w_dists = {}
		if word in self.word_vectors:
			all_vectors = (self.word_vectors,)
			for clue in self.cm_cluelist:
				b_dist = cos_dist(self.concatenate(clue, all_vectors), self.concatenate(word, all_vectors))
				nn_w_dists[clue] = b_dist
		else:
			if self.configuration.verbose:
				print(word, "not in word_vectors")
		self.word_dists[word] = nn_w_dists
		return {k: 1-v for k,v in nn_w_dists.items()}

	def get_clue(self, blue_words, red_words, chosen_blue_words):
		bests = {}

		best_clue = ''
		best_blue_dist = np.inf
		for clue in self.cm_cluelist:
			if not self.arr_not_in_word(clue, blue_words.union(red_words)):
				continue

			red_dist = np.inf
			for red_word in red_words:
				if clue in self.word_dists[red_word]:
					if self.word_dists[red_word][clue] < red_dist:
						red_dist = self.word_dists[red_word][clue]
						if self.configuration.use_heuristics:
							dict2vec_dist = 0.2*utils.get_dict2vec_dist(red_word, clue)
							#print(red_word, clue, "red dist", red_dist, "dict2vec penalty", dict2vec_penalty)
							red_dist += dict2vec_dist
							if self.configuration.debug_file:
								with open(self.configuration.debug_file, 'a') as f:
									f.write(" ".join([str(x) for x in [
										"\n","red word:", red_word, "clue", clue, "red dist", red_dist, "dict2vec dist", dict2vec_dist
									]]))
			worst_blue_dist = 0
			for blue in chosen_blue_words:
				if clue in self.word_dists[blue]:
					dist = self.word_dists[blue][clue]
					if self.configuration.use_heuristics:
						dict2vec_dist = 0.5*utils.get_dict2vec_dist(blue, clue)
						#print(blue, clue, "blue dist", dist, "dict2vec score", dict2vec_score)
						dist += dict2vec_dist
						if self.configuration.debug_file:
								with open(self.configuration.debug_file, 'a') as f:
									f.write(" ".join([str(x) for x in [
										"\n","blue word:", blue, "clue", clue, "blue dist", dist, "dict2vec dist", dict2vec_dist
									]]))
				else:
					dist = np.inf
				if dist > worst_blue_dist:
					worst_blue_dist = dist


			if worst_blue_dist < best_blue_dist and worst_blue_dist < red_dist:
				best_blue_dist = worst_blue_dist
				best_clue = clue
				# print(worst_blue_dist,chosen_blue_words,clue)
		chosen_num = len(chosen_blue_words)
		bests[chosen_num] = (chosen_blue_words, best_clue, best_blue_dist)

		chosen_clue_info = bests[chosen_num]
		for clue_num, clue_info in bests.items():
			best_blue_words, best_clue, worst_blue_dist = clue_info
			if worst_blue_dist != -np.inf:
				# print(worst_blue_dist, chosen_clue_info, chosen_num)
				chosen_clue_info = clue_info
				chosen_num = clue_num
		if self.configuration.debug_file:
			with open(self.configuration.debug_file, 'a') as f:
				f.write(" ".join([str(x) for x in [
					"\n",'chosen_clue_info is:', chosen_clue_info
				]]))
		return chosen_clue_info[1], chosen_clue_info[2]

	def arr_not_in_word(self, word, arr):
		if word in arr:
			return False
		lemm = self.wordnet_lemmatizer.lemmatize(word)
		lancas = self.lancaster_stemmer.stem(word)
		for i in arr:
			if i == lemm or i == lancas:
				return False
			if i.find(word) != -1:
				return False
			if word.find(i) != -1:
				return False
		return True

	def concatenate(self, word, wordvecs):
		concatenated = wordvecs[0][word]
		for vec in wordvecs[1:]:
			concatenated = np.hstack((concatenated, vec[word]))
		return concatenated

	def dict2vec_embedding_weight(self):
		return 1.0
