import random
from numpy.linalg import norm
import numpy as np

vocabulary_file='word_embeddings.txt'

# Read words
print('Read words...')
with open(vocabulary_file, 'r', encoding='utf-8') as f:
    words = [x.rstrip().split(' ')[0] for x in f.readlines()]

# Read word vectors
print('Read word vectors...')
with open(vocabulary_file, 'r', encoding='utf-8') as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = [float(x) for x in vals[1:]]

vocab_size = len(words)
vocab = {w: idx for idx, w in enumerate(words)}
ivocab = {idx: w for idx, w in enumerate(words)}

# Vocabulary and inverse vocabulary (dict objects)
print('Vocabulary size')
print(len(vocab))
print(vocab['man'])
print(len(ivocab))
print(ivocab[10])

# W contains vectors for
print('Vocabulary word vectors')
vector_dim = len(vectors[ivocab[0]])
W = np.zeros((vocab_size, vector_dim))
for word, v in vectors.items():
    if word == '<unk>':
        continue
    W[vocab[word], :] = v
print(W.shape)

# Function to compute Euclidean distance
def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


# Function to find the 3 most similar words by distance
def find_similar_words(word, W, vocab, ivocab):
    if word not in vocab:
        # Return the error message if the word is not found in the vocabulary
        return f"{word} not found in the vocabulary!"
    
    word_vector = W[vocab[word]]
    distances = np.array([euclidean_distance(word_vector, W[i]) for i in range(W.shape[0])])
    
    # Exclude the input word by setting its distance to a large value (infinity)
    word_idx = vocab[word]
    distances[word_idx] = np.inf  # Set a very high distance for the input word
    
    # Get indices of the top 3 closest words (smallest distance)
    top_indices = distances.argsort()[:3]  # Ascending order (smallest distances)
    
    return [(ivocab[idx], distances[idx]) for idx in top_indices]



# Main loop for analogy
while True:
    input_term = input("\nEnter word (EXIT to break): ")
    if input_term == 'EXIT':
        break
    else:
        # Similar words search using Euclidean distance
        similar_words = find_similar_words(input_term, W, vocab, ivocab)
        # If the function returns an error message, print it
        if isinstance(similar_words, str):
            print(similar_words)
        else:
            print("\n                               Word       Distance\n")
            print("---------------------------------------------------------\n")
            for word, distance in similar_words:
                print(f"{word:>35}\t\t{distance:.6f}\n")

        

