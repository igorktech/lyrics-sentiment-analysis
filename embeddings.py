import torch
import json
import numpy as np

EMBEDDING_DIMS = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W1 = torch.load('content/embedding_weights.pt', map_location=torch.device(device))
vocabulary = torch.load('content/vocabulary.pkl')
with open('content/freq.json', 'r') as f:
    frequency = json.load(f)
def compute_sentence_embedding(sentence, model=W1, vocab=vocabulary,freq = frequency):
    # Tokenize the sentence into words
    words = sentence.split()

    # Get the embeddings for each word in the sentence
    word_embeddings = []
    for word in words:
        try:
            word_embeddings.append(model[:, vocab.index(word)].detach().numpy()*frequency[word])
        except:
            word_embeddings.append(np.zeros(EMBEDDING_DIMS))

    # Compute the sentence embedding as the mean of the word embeddings
    if word_embeddings:
        sentence_embedding = np.mean(word_embeddings, axis=0)
    else:
        sentence_embedding = np.zeros(EMBEDDING_DIMS)

    return sentence_embedding