from torch import nn
import sigmoid_bce_loss

class SkipGram(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, id2word, word2frequency, neg_sampling=False) -> None:
        super(SkipGram, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(self.vocabulary_size, self.embedding_dim)

        if neg_sampling:
            self.output_weights = nn.Embedding(vocabulary_size, embedding_dim)
            self.loss_function = sigmoid_bce_loss.SigmoidBCELoss(id2word, word2frequency)
            self.loss_function.output_weights = self.output_weights
        else:
            self.output_weights = nn.Linear(self.embedding_dim, self.vocabulary_size)
            self.loss_function = nn.CrossEntropyLoss()

    def forward(self, input_idx, target_idx=None):
        input_embeddings = self.embeddings(input_idx)

        if isinstance(self.loss_function, sigmoid_bce_loss.SigmoidBCELoss):
            if target_idx is None:
                raise ValueError("target_idx must be provided when using negative sampling.")
            return self.loss_function.forward(input_embeddings, self.output_weights, target_idx)
        else:
            output_logits = self.output_weights(input_embeddings)
            return output_logits
