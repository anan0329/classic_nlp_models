import torch
from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, filter_size, num_filter, num_class):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1d = nn.Conv1d(embedding_dim, num_filter, filter_size, padding=1)
        self.relu = F.relu
        self.linear = nn.Linear(num_filter, num_class)
        
    def forward(self, inputs):
        embedding = self.embedding(inputs)
        convolution = self.relu(self.conv1d(embedding.permute(0,2,1)))
        pooling = F.max_pool1d(convolution, kernel_size=convolution.shape[2])
        output = self.linear(pooling.squeeze(dim=2))
        log_probs = F.log_softmax(output, dim=1)
        return log_probs
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    inputs = torch.randint(0, 10, (32,10)).to(device)
    model = CNN(vocab_size=10, embedding_dim=256, filter_size=3, num_filter=100, num_class=2).to(device)
    out = model(inputs)
    print(out.shape)