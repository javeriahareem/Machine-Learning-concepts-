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

# Function to find the nearest neighbors using Euclidean distance
def find_nearest_neighbors(vec, W, excluded_words, top_k=2):
    # Calculate Euclidean distance between vec and all word vectors in W
    distances = np.linalg.norm(W - vec, axis=1)
    
    # Sort distances and get indices, but exclude input words
    nearest_indices = [i for i in distances.argsort() if ivocab[i] not in excluded_words][:top_k]
    
    return nearest_indices, distances[nearest_indices]

# Main loop for analogy
while True:
    input_term = input("\nEnter analogy in format: 'word1-word2-word3' (EXIT to break): ")
    if input_term == 'EXIT':
        break

    try:
        # Split the input into three words using hyphen
        word1, word2, word3 = input_term.lower().split('-')

        # Get vectors for word1, word2, word3
        vec1 = np.array(vectors[word1])
        vec2 = np.array(vectors[word2])
        vec3 = np.array(vectors[word3])

        # Compute the target vector: vec3 + (vec2 - vec1)
        target_vec = vec3 + (vec2 - vec1)

        # Find the nearest neighbors to the target vector, excluding the input words
        excluded_words = {word1, word2, word3}  # Create a set of input words to exclude
        nearest_indices, nearest_distances = find_nearest_neighbors(target_vec, W, excluded_words, top_k=2)

        print("\n                               Word       Distance\n")
        print("---------------------------------------------------------\n")
        for i, idx in enumerate(nearest_indices):
            print(f"{ivocab[idx]:>35}\t\t{nearest_distances[i]:.6f}\n")  # Print distance

    except KeyError as e:
        print(f"Word '{e.args[0]}' not found in the vocabulary. Please try again.")
    except ValueError:
        print("Invalid input format. Please use 'word1-word2-word3' format.")