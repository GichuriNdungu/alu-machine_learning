#!/usr/bin/env python3
"""function that creates a bag of words embedding matrix"""

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """creates a bag of words embedding matrix
    params:
        Vocab: []: words within sentence to use for analysis
        sentences: []: sentences to use for analysis
    return:
        embeddings: np.array(s, f), where s = number of sentences in params
                    f = number of features analyzed
        Features: [] : Names of features used for analysis"""

    if vocab:
        vectorizer = CountVectorizer(vocabulary=vocab)
    else:
        vectorizer = CountVectorizer()
    X_train_counts = vectorizer.fit_transform(sentences)
    embeddings = X_train_counts.toarray()
    features = vectorizer.get_feature_names()
    return embeddings, features
