from annoy import AnnoyIndex
from bert_embedding import BertEmbedding
import numpy as np
import re
from nltk.corpus import brown
#import mxnet as mx

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

def create_graph():
    #ctx = mx.gpu(0)
    regex = re.compile('[^a-zA-Z]')
    bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_cased')
    emb_size = 768
    t = AnnoyIndex(emb_size, metric='angular')
    tree_idx = 0
    tree_idx_to_word_dict = dict()
    words = []


    for word in brown.words():
        # process word and insert into our growing array of words
        word = regex.sub('', word.lower())
        if word == '' or word in stop_words:
            continue
        words.append(word)

    print("PRE-PROCESSED BROWN CORPUS")

    # create bert token embeddings.
    embeddings = bert_embedding(words, 'avg')

    print("CREATED BERT EMBEDDINGS")

    mod = 50000
    # add the embeddings to the annoy tree, and add to our mapping of tree indices -> words
    for i in range(len(embeddings)):
        if (i % mod == 0): print("ADDED ", i, " EMBEDDINGS TO ANNOY TREE")
        emb_vector = embeddings[i][1][0]

        word = words[i]
        # print ("word, emb vector, tree idx", word, emb_vector, tree_idx)
        tree_idx_to_word_dict[tree_idx] = word
        tree_idx += 1
        t.add_item(tree_idx, emb_vector)

    t.build(100)

    t.save('annoy_tree_bert_emb_768_brown_corpus.ann')
    np.save('annoy_tree_index_to_word_bert_emb_768_brown_corpus.npy', tree_idx_to_word_dict)

if __name__=='__main__':
    create_graph()
