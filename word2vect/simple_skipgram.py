from torch import nn

class SimpleSkipGram(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim) -> None:
        super(SimpleSkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocabulary_size, embedding_dim)
        self.output_weights = nn.Linear(embedding_dim, vocabulary_size)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, input_idx, target_idx=None):
        input_embeddings = self.embeddings(input_idx)
        output_logits = self.output_weights(input_embeddings)
        return output_logits
