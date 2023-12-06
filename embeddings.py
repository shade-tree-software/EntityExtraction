import pdb
import pickle
import string
import sys

import nltk
import numpy as np
from nltk.corpus import stopwords

from utils import (cosine_similarity, process_text)

# nltk.download('stopwords')

# cd embeddings
# wget https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
# unzip -q glove.6B.zip

# en_embeddings_subset = pickle.load(open("embeddings/en_embeddings.p", "rb"))

# en_embeddings = {}
# with open("embeddings/glove.6B.300d.txt") as f:
#     for line in f:
#         word, coefs = line.split(maxsplit=1)
#         coefs = np.fromstring(coefs, "f", sep=" ")
#         en_embeddings[word] = coefs
# pickle.dump(en_embeddings, open("embeddings/glove.6B.300d.p", "wb"))

EMBEDDINGS_FILE = "embeddings/glove.6B.300d.p"
en_embeddings = pickle.load(open(EMBEDDINGS_FILE, "rb"))

input_file = sys.argv[1]

def nearest_neighbor(v, candidates, k=1, cosine_similarity=cosine_similarity):
    """
    Input:
      - v, the vector you are going find the nearest neighbor for
      - candidates: a set of vectors where we will find the neighbors
      - k: top k nearest neighbors to find
    Output:
      - the indices of the top k closest vectors
    """
    cos_similarities = []
    # get cosine similarity of input vector v and each candidate vector
    for candidate in candidates:
        cos_similarities.append(cosine_similarity(v, candidate))
    # sort the similarity list and get the k most similar indices    
    return np.flip(np.argsort(cos_similarities))[:k]

def get_document_embedding(text, embeddings, process_text=process_text):
    '''
    Input:
        - text: a string
        - en_embeddings: a dictionary of word embeddings
    Output:
        - doc_embedding: sum of all word embeddings in the tweet
    '''
    doc_embedding = np.zeros(300)
    print(f"initial word count: {len(text.split())}")
    # process the document into a list of words (process the tweet)
    processed_doc = process_text(text)
    print(f"word count after processing: {len(processed_doc)}")
    words_with_embeddings = set()
    for word in processed_doc:
        if word not in ['transfer', 'type', 'html', 'utf', 'content', 'text', 'div', 'http', 'www', 'org']:
            # add the word embedding to the running total for the document embedding
            word_embedding = embeddings.get(word, 0)
            if isinstance(word_embedding, np.ndarray):
                words_with_embeddings.add(word)
            doc_embedding += word_embedding
    print(f"words with embeddings: {' '.join(list(words_with_embeddings))}")
    return doc_embedding

with open(input_file, "r") as f:
    text = f.read()

embedding = get_document_embedding(text, en_embeddings)
