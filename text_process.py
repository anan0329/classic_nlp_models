from collections import defaultdict
from typing import List

class Vocab:
    def __init__(self, tokens: List[str]=None):
        self.idx_to_token = list()
        self.token_to_idx = dict()
        
        if tokens:
            if "<unk>" not in tokens:
                tokens = tokens + ["<unk>"]
                
            self.token_to_idx = {token: idx for idx, token in enumerate(tokens)}
            self.idx_to_token = [*self.token_to_idx]
            # for token in tokens:
            #     self.idx_to_token.append(token)
            #     self.token_to_idx[token] = len(self.token_to_idx) - 1
            self.unk = self.token_to_idx['<unk>']
            
    @classmethod
    def build(cls, text, min_freq=1, reserved_tokens=None):
        token_freqs = defaultdict(int)
        for sent in text:
            for token in sent:
                token_freqs[token] += 1
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token, freq in token_freqs.items() if freq >= min_freq and token != "<unk>"]
        
        return cls(uniq_tokens)
    
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.unk)
    
    def convert_tokens_to_ids(self, tokens):
        return [self.token_to_idx[token] for token in tokens]
    
    def convert_ids_to_tokens(self, indices):
        return [self.idx_to_token[index] for index in indices]
    
if __name__ == "__main__":
    sents = [["I", "see", ",", "I", "come", ",", "I", "conquer"], 
             ["A", "friend", "in", "need", "is", "a", "friend", "indeed"]]
    print(f"{sents[0]}=")
    vocab = Vocab.build(sents)
    print(vocab.convert_tokens_to_ids(sents[0]))
    print(vocab.convert_ids_to_tokens(range(len(vocab))))