import re
import os
import string
import pickle
from scipy.spatial import distance

punctuation = re.compile("[" + re.escape(string.punctuation) + "]")

word_to_dict2vec_embeddings = None

def remove_punctuation(word):
	return word.translate(str.maketrans('', '', string.punctuation))

def is_single_word(word):
	return len(neighbor.split("_")) == 1 and len(neighbor.split("-")) == 1


def get_dict2vec_score(chosen_words, potential_clue, red_words):
	"""
	:param chosen_words: the board words intended for the potential clue
	:param potential_clue: potential candidate clue
	:param red_words: the red words on the board
	returns: the similarity of the two input embedding vectors using their cosine distance
	"""
	global word_to_dict2vec_embeddings
	if word_to_dict2vec_embeddings == None:
		try:
			input_file = open('data/word_to_dict2vec_embeddings','rb')
		except IOError:
		  print("Error: data/word_to_dict2vec_embeddings does not exist.")
		  return 0.0
		word_to_dict2vec_embeddings = pickle.load(input_file)


	if potential_clue not in word_to_dict2vec_embeddings:
		return 0.0

	potential_clue_embedding = word_to_dict2vec_embeddings[potential_clue]
	dict2vec_similarities = []
	for chosen_word in chosen_words:
		if chosen_word in word_to_dict2vec_embeddings:
			chosen_word_embedding = word_to_dict2vec_embeddings[chosen_word]
			dict2vec_similarities.append(get_dict2vec_similarity(chosen_word_embedding, potential_clue_embedding))

	max_red_similarity = float("-inf")
	for red_word in red_words:
		if red_word in word_to_dict2vec_embeddings:
			red_word_embedding = word_to_dict2vec_embeddings[red_word]
			red_word_similarity = get_dict2vec_similarity(red_word_embedding, potential_clue_embedding)
			if red_word_similarity > max_red_similarity:
					max_red_similarity = red_word_similarity

	return sum(dict2vec_similarities) - 0.5 * max_red_similarity

def get_dict2vec_similarity(word_embedding_0, word_embedding_1):
	"""
	:param word_embedding_0: dict2vec word embedding 0
	:param word_embedding_1: dict2vec word embedding 1
	returns: the similarity of the two input embedding vectors using their cosine distance
	"""
	cosine_distance = distance.cosine(word_embedding_0, word_embedding_1)
	# scipy cosine distance calculation is 1-(cos(u,v))
	return 1.0 if cosine_distance == 0 else  (1 - cosine_distance)

def get_dict2vec_dist(word1, word2):
	global word_to_dict2vec_embeddings
	if word_to_dict2vec_embeddings == None:
		print("reloading embedding")
		try:
			input_file = open('data/word_to_dict2vec_embeddings','rb')
		except IOError:
		  print("Error: data/word_to_dict2vec_embeddings does not exist.")

		word_to_dict2vec_embeddings = pickle.load(input_file)

	try:
		return distance.cosine(word_to_dict2vec_embeddings[word1], word_to_dict2vec_embeddings[word2])
	except KeyError:
		return 1.0
