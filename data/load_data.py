from .text_process import Vocab
from nltk.corpus import sentence_polarity
from sklearn.model_selection import train_test_split
import numpy as np


class SentencePolarity:
    def __init__(self, test_prop: float = 0.8) -> None:
        self.test_prop = test_prop
        self.train_sents = None
        self.test_sents = None
        self.train_trg = None
        self.test_trg = None
        self.vocab = None
        self.train_x = None
        self.test_x = None
        
    def load(self):
        pos_sents = sentence_polarity.sents(categories="pos")
        neg_sents = sentence_polarity.sents(categories="neg")
        
        train_pos_sents, test_pos_sents = train_test_split(pos_sents, test_size=self.test_prop)
        train_neg_sents, test_neg_sents = train_test_split(neg_sents, test_size=self.test_prop)
        
        self.train_sents = train_pos_sents + train_neg_sents
        self.test_sents = test_pos_sents + test_neg_sents
        self.train_trg = [1] * len(train_pos_sents) + [0] * len(train_neg_sents)
        self.test_trg = [1] * len(test_pos_sents) + [0] * len(test_neg_sents)
        
        return True
    
    def process(self):
        
        load = self.load()
        
        assert load is not None, 'load function must return True'
        
        all_sents = self.train_sents + self.test_sents
        self.vocab = Vocab.build(all_sents)
        
        self.train_x = [self.vocab.convert_tokens_to_ids(sent) for sent in self.train_sents]
        self.test_x = [self.vocab.convert_tokens_to_ids(sent) for sent in self.test_sents]
        


if __name__ == "__main__":
    
    sentence_polarity_data = SentencePolarity(test_prop=0.8)
    sentence_polarity_data.process()
    
    
