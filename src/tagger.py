from abc import ABC, abstractmethod

import nltk


class AbstractTagger(ABC):
    @abstractmethod
    def __init__(self):
        self._possibleTagsList = None

    @property
    def possible_tags(self):
        return self._possibleTagsList

    @abstractmethod
    def map_sentence(self, sentence):
        raise NotImplementedError()


class UniversalTagger(AbstractTagger):
    def __init__(self):
        AbstractTagger.__init__(self)
        self._possibleTagsList = {'ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X',
                                  'EMPTY'}

    def map_sentence(self, sentence):
        list_of_pos_tags = nltk.pos_tag(sentence, tagset='universal')
        list_of_tags = [None] * len(list_of_pos_tags)
        for i in range(len(list_of_pos_tags)):
            list_of_tags[i] = 'EMPTY' if sentence[i] == '-' else list_of_pos_tags[i][1]
        return list_of_tags


class FullTagger(AbstractTagger):
    def __init__(self):
        AbstractTagger.__init__(self)
        self._possibleTagsList = {'VBN', 'VBZ', 'VBG', 'VBP', 'VBD', 'MD', 'NN', 'NNPS', 'NNP', 'NNS', 'JJS', 'JJR',
                                  'JJ', 'RB', 'RB', 'RB', 'EMPTY', 'CD', 'IN', 'PDT', 'CC', 'EX', 'POS', 'RP', 'FW',
                                  'DT', 'UH', 'TO', 'PRP', 'PRP$', '$', 'WP', 'WP$', 'WDT', 'WRB'}

    def map_sentence(self, sentence):
        list_of_pos_tags = nltk.pos_tag(sentence)
        list_of_tags = [None] * len(list_of_pos_tags)
        for i in range(len(list_of_pos_tags)):
            tag = list_of_pos_tags[i][1]
            list_of_tags[i] = tag if tag in self._possibleTagsList else 'OTHER'
        return list_of_tags
