from annoy import AnnoyIndex
from bert_embedding import BertEmbedding
import numpy as np
import re
from nltk.corpus import wordnet as wn
from nltk.corpus import brown


EOD_token = "---end.of.document---"
doc_limit = 400
stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
#file_path = "/usr/project/xtmp/dk160/bert/wiki_2012.txt"
file_path = "/Users/divyakoyyalagunta/projects/multisense/data/wiki_2012_data/wiki_2012.txt"
codewords_path = "/Users/divyakoyyalagunta/projects/multisense/codenames/codewords.txt"
def create_graph():
    #ctx = mx.gpu(0)
    regex = re.compile('[^a-zA-Z]')
    bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_cased')
    emb_size = 768
    t = AnnoyIndex(emb_size, metric='angular')
    tree_idx = 0
    tree_idx_to_word_dict = dict()
    doc_num = 0
    words = []

    with open(file_path) as file:

        for i, line in enumerate(file):

            if doc_num == doc_limit:
                break

            words_on_line = line.strip().lower().split()

            for word in words_on_line:

                if word == EOD_token:

                    # go through this document's words, and create bert token embeddings.
                    # add the embeddings to the annoy tree, and add to our mapping of tree indices -> words
                    embeddings = bert_embedding(words, 'avg')

                    for i in range(len(embeddings)):
                        emb_vector = embeddings[i][1][0]

                        word = words[i]
                        #print ("word, emb vector, tree idx", word, emb_vector, tree_idx)
                        tree_idx_to_word_dict[tree_idx] = word
                        tree_idx += 1
                        t.add_item(tree_idx, emb_vector)
                    print("completed document ", doc_num)
                    # update variables
                    words = []
                    doc_num += 1


                else:
                    # process word and insert into our growing array of words for the current document
                    word = regex.sub('', word.lower)
                    # TODO: remove stop words
                    if word == '' or word in stop_words:
                        continue
                    words.append(word)
    t.build(100)
    t.save('bert.ann')
    np.save('annoy_tree_index_to_word.npy', tree_idx_to_word_dict)

def create_graph_using_wordnet():

    regex = re.compile('[^a-zA-Z]')
    bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_cased')
    emb_size = 768
    t = AnnoyIndex(emb_size, metric='angular')
    tree_idx = 0
    tree_idx_to_word_dict = dict()
    words = []

    with open(codewords_path) as file:

        for i, line in enumerate(file):
            codeword = line.strip().lower()
            print("CODEWORD", codeword)
            for synset in wn.synsets(codeword):
                print("synset", synset.name())
                for example in synset.examples():
                    print("example", example)
                    for word in example.split():
                        word = regex.sub('', word.lower())
                        if word not in stop_words:
                            print("example word", word)
                            words.append(word)

    print("num words", len(words))

    print("---------done processing words---------")
    embeddings = bert_embedding(words, 'avg')
    print("num embeddings", len(embeddings))

    print("---------adding embeddings to tree---------")

    for i in range(len(embeddings)):
        if(len(embeddings[i]) >= 2 and len(embeddings[i][1]) >= 1):
            emb_vector = embeddings[i][1][0]

            word = words[i]
            #print ("word, emb vector, tree idx", word, emb_vector, tree_idx)
            tree_idx_to_word_dict[tree_idx] = word
            tree_idx += 1
            t.add_item(tree_idx, emb_vector)
        else:
            print(i)
            print("bad embedding", embeddings[i])

    print("---------building tree---------")
    t.build(100)
    t.save('bert_wordnet.ann')
    np.save('annoy_tree_index_to_word_bert_wordnet.npy', tree_idx_to_word_dict)

if __name__=='__main__':
    #create_graph()
    create_graph_using_wordnet()
