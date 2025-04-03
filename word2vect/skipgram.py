from torch import nn
import sigmoid_bce_loss

class SkipGram(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, id2word, word2frequency) -> None:
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocabulary_size, embedding_dim)
        self.output_weights = nn.Embedding(vocabulary_size, embedding_dim)
        self.loss_function = sigmoid_bce_loss.SigmoidBCELoss(id2word, word2frequency)
        self.loss_function.output_weights = self.output_weights

    def forward(self, input_idx, target_idx):
        input_embeddings = self.embeddings(input_idx)
        return self.loss_function.forward(input_embeddings, self.output_weights, target_idx)
