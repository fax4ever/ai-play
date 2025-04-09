from torch import nn
from sigmoid_bce_loss import SigmoidBCELoss

class SkipGram(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, id2word, word2frequency) -> None:
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocabulary_size, embedding_dim)
        self.output_weights = nn.Embedding(vocabulary_size, embedding_dim)
        self.loss_function = SigmoidBCELoss(id2word, word2frequency)

    def forward(self, input_idx, target_idx):
        input_embeddings = self.embeddings(input_idx)
        return self.loss_function(input_embeddings, self.output_weights, target_idx)
