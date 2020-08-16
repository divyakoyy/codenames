from annoy import AnnoyIndex
from bert_embedding import BertEmbedding
import numpy as np
import re
from gensim.models import KeyedVectors

def get_word_emb(bert_embedding, word):
    '''
    returns a single word embedding
    '''
    try:
        emb = bert_embedding([word], 'avg')[0][1][0]  # first word, first vector array, array itself
        normalized_emb = emb/np.linalg.norm(emb)
        return normalized_emb
    except:
        print(word, "does not have a bert embedding")
        return []

def create_graph():
    #ctx = mx.gpu(0)
    regex = re.compile('[^a-zA-Z]')
    bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_cased')
    emb_size = 768
    t = AnnoyIndex(emb_size, metric='angular')
    tree_idx = 0
    tree_idx_to_word_dict = dict()
    glove_model = KeyedVectors.load_word2vec_format('../data/glove-wiki-gigaword-300.txt.gz')
    print("Vocab size", glove_model.vocab.size)
    i = 0
    mod = 50000
    for word in glove_model.vocab:
        if (i % mod == 0): print("ADDED ", i, " EMBEDDINGS TO ANNOY TREE")
        
        emb_vector = get_word_emb(bert_embedding, word)

        if len(emb_vector) == 0: 
            continue
        # print ("word, length of emb vector, sum of emb vector, tree idx", word, len(emb_vector), np.linalg.norm(emb_vector), tree_idx)
        # add the embeddings to the annoy tree, and add to our mapping of tree indices -> words
        tree_idx_to_word_dict[tree_idx] = word
        tree_idx += 1
        t.add_item(tree_idx, emb_vector)

        i += 1


    t.build(100)

    t.save('annoy_tree_bert_emb_768_wiki_news_corpus_single_word.ann')
    np.save('annoy_tree_index_to_word_bert_emb_768_wiki_news_single_word.npy', tree_idx_to_word_dict)

if __name__=='__main__':
    create_graph()
