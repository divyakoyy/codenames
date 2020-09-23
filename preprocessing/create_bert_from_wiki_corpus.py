from annoy import AnnoyIndex
from bert_embedding import BertEmbedding
import numpy as np
import re
from gensim.models import KeyedVectors
from gensim.models.word2vec import Text8Corpus

stopwords = set([
    'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'get', 'put',
])

# (averaged_embedding, count_of_word) at index i in embedding_vector_averages is the average BERT embedding of a word. 
# The word to index map is stored in word_to_idx_dict
embedding_vector_averages = []
idx = 0
word_to_idx_dict = dict()
bank_idx = 1

def add_sentence_emb(bert_embedding, sentence):
    '''
    returns a single word embedding
    '''
    global idx
    global word_to_idx_dict
    global embedding_vector_averages
    global bank_idx

    try:
        emb = bert_embedding([sentence], 'avg')
        words = emb[0][0]
        embedding_vectors = emb[0][1]

        # emb = bert_embedding([word], 'avg')[0][1][0]  # first word, second vector array, array itself
        # normalized_emb = emb/np.linalg.norm(emb)
        # return normalized_emb

        for (word, embedding) in zip(words, embedding_vectors):
            # if word == 'bank':
            #     embedding = [bank_idx]*768
            #     bank_idx += 2
            if word in stopwords:
                continue
            # First time we've seen this word
            if word not in word_to_idx_dict:
                word_to_idx_dict[word] = idx
                embedding_vector_averages.append((embedding, 1))
                idx += 1
            # Otherwise, average this embedding with the running average
            else:

                curr_word_idx = word_to_idx_dict[word]
                curr_average_vector = embedding_vector_averages[curr_word_idx][0]
                curr_count = embedding_vector_averages[curr_word_idx][1]

                averaged_embedding = np.average( np.array([ curr_average_vector, embedding ]), axis=0, weights = [curr_count, 1] )
                embedding_vector_averages[curr_word_idx] = (averaged_embedding, curr_count + 1)
                # if word == 'bank':
                #     print("curr_word_idx", curr_word_idx, "curr_average_vector", curr_average_vector[:10], "curr_count", curr_count)
                #     print("new embedding", embedding[:10])
                #     print("new averaged embedding", averaged_embedding[:10])


    except:
        print("Could not get bert embedding for sentence", sentence)

def create_graph():
    #ctx = mx.gpu(0)
    bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_cased')
    
    text8Corpus = Text8Corpus("/Users/divyakoyyalagunta/Projects/codenames/text8.txt", max_sentence_length=10)

    # sentences = ["urged Filipinos to stop weeping for the man who had laughed all the way to the bank",
    #  "Soon after setting off we came to a forested valley along the bank of the Gwaun",
    #   "The condom balloon was denied official entry status this year",
    #   "The marine said, get down behind that grass bank, sir, and he immediately lobbed a mills grenade into the river"]
    for sentence in text8Corpus:
        joined_sentence = (" ").join(sentence)

        add_sentence_emb(bert_embedding, joined_sentence)

    print("Number of embeddings", len(embedding_vector_averages))
    print("Number of words in word_to_idx_dict", len(word_to_idx_dict.keys()))

    tree_idx = 0
    mod = 50000
    emb_size = 768
    t = AnnoyIndex(emb_size, metric='angular')

    for x in range(len(embedding_vector_averages)):
        if (tree_idx % mod == 0): print("ADDED ", tree_idx, " EMBEDDINGS TO ANNOY TREE")

        embedding = embedding_vector_averages[x][0]

        if len(embedding) == 0: 
            continue

        t.add_item(x, embedding)

        tree_idx += 1

    t.build(100)

    idx_to_word_dict = {v: k for k, v in word_to_idx_dict.items()}

    t.save('annoy_tree_bert_emb_768_test.ann')
    np.save('annoy_tree_index_to_word_bert_emb_768_test.npy', idx_to_word_dict)

if __name__=='__main__':
    create_graph()
