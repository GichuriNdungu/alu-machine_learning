#!/usr/bin/env python3
"""function that creates a TF_Idf embedding matrix"""

from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, window=5, negative=5, cbow=True, iterations=5, seed=0, workers=1):
    """ params:
            sentences: list of sentences to be trained on
            size = dimensionality of the embedding layer
            min_count = minimum number of occurrences of a word for use in training
            window: the maximum distance between the current and predicted word within a sentence
            negative is the size of negative sampling
            cbow is a boolean to determine the training type; True is for CBOW; False is for Skip-gram
            iterations is the number of iterations to train over
            seed is the seed for the random number generator
            workers is the number of worker threads to train the model
        return: 
            trained model"""
    if cbow:
        cbow_flag = 0
    else:
        cbow_flag = 1
    model = FastText(sentences=sentences,
                     size=size,
                     min_count=min_count,
                     window=window,
                     negative=negative,
                     sg=cbow_flag,
                     iter=iterations,
                     seed=seed,
                     workers=workers)
    model.train(sentences,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    return model

