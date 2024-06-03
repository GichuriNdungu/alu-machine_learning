#!/usr/bin/env python3
'''function that calculates the bleu score of a sentence'''
import numpy as np


def uni_bleu(references, sentence):
    '''params:
            references: list of reference translations
                        each list consists of words in each reference
            sentence: list with the model's proposed sentence'''
    unique_words = list(set(sentence))
    max_counts = {}
    for word in unique_words:
        # count its max in the reference
        max_count = 0
        for reference in references:
            count = reference.count(word)
            if count > max_count:
                max_count = count
        max_counts[word] = max_count
    total_words = 0
    for word in sentence:
        total_words += 1
    clipped = sum(max_counts.values())
    prec = clipped/total_words
    r = min(len(ref) for ref in references)
    c = len(sentence)
    brevity_penalty = np.exp(1 - r/c) if c < r else 1
    bleu_score = brevity_penalty * prec
    return bleu_score
