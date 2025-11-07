import os
import sys
import spacy
from spacy_syllables import SpacySyllables


print('loading syllables...')
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("syllables", after="tagger")
print('yay syllables!')


def count_total_syllables(text):
    doc = nlp(text)

    total_syllables = 0
    for token in doc:
        if token.is_punct or token.is_space or (token._.syllables_count is None):
            continue

        total_syllables += token._.syllables_count

    return total_syllables


if __name__ == '__main__':
    text = 'Hello, the quick brown fox jumps over the lazy dog because the precipitation in Spain stays mainly in the plain'
    print(count_total_syllables(text))
    for word in text.split():
        print((word, count_total_syllables(word)))
