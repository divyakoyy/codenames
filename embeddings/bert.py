
from annoy import AnnoyIndex
import numpy as np

class Bert(object):

	def __init__(self, configuration=None):


		# Initialize variables
		self.configuration = configuration
		self.graphs = dict() #TODO: visualizations for other embedding methods

		emb_size = 768 # bert_12_768_12, book_corpus_wiki_en_cased
		self.bert_annoy_tree = AnnoyIndex(emb_size, 'angular')
		self.bert_annoy_tree.load('data/annoy_tree_bert_emb_768_brown_corpus.ann')
		self.bert_annoy_tree_idx_to_word = np.load('data/annoy_tree_index_to_word_bert_emb_768_brown_corpus.npy', allow_pickle=True).item()
		self.bert_annoy_tree_word_to_idx = {v: k for k, v in self.bert_annoy_tree_idx_to_word.items()}

	"""
	Required codenames methods
	"""

	def get_weighted_nn(self, word, n=500):
		nn_w_similarities = dict()
		
		if word not in self.bert_annoy_tree_word_to_idx:
			return nn_w_similarities

		annoy_idx = self.bert_annoy_tree_word_to_idx[word]
		neigbors_and_distances = self.bert_annoy_tree.get_nns_by_item(annoy_idx, n, include_distances=True)

		for neighbor_annoy_idx, distance in zip(neigbors_and_distances[0], neigbors_and_distances[1]):
			neighbor_word = self.bert_annoy_tree_idx_to_word[neighbor_annoy_idx].lower()
			if len(neighbor_word.split("_")) > 1 or len(neighbor_word.split("-")) > 1:
				continue
			
			similarity = 1.0 if distance == 0.0 else (1 - distance/2)
			if neighbor_word not in nn_w_similarities:
				nn_w_similarities[neighbor_word] = similarity
			nn_w_similarities[neighbor_word] = max(similarity, nn_w_similarities[neighbor_word])

		return {k: v for k, v in nn_w_similarities.items() if k != word}

	def rescale_score(self, chosen_words, potential_clue, red_words):
		"""
		:param chosen_words: potential board words we could apply this clue to
		:param clue: potential clue
		:param red_words: opponent's words
		returns: penalizes a potential_clue for being have high word2vec similarity with opponent's words
		"""
		# TODO

		return 0

	def dict2vec_embedding_weight(self):
		return 2.0
