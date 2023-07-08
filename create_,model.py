import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader.bag_of_words import BagOfWordDataset, collate_fn
from preprocess.load_data import SentencePolarity

EMBEDDING_DIM = 256
HIDDEN_DIM = 256
NUM_CLASS = 2
BATCH_SIZE = 32
NUM_EPOCH = 10


if __name__ == "__main__": 
    sp_data = SentencePolarity(test_prop=0.8)
    sp_data.process()
    
    train_dataset = BagOfWordDataset(sp_data.train_x, sp_data.train_trg)
    test_dataset = BagOfWordDataset(sp_data.test_x, sp_data.test_trg)
    
    # print(sp_data.train_sents[100])
    # print(sp_data.test_sents[100])
    # print(train_dataset[100])
    # print(test_dataset[100])
    
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')