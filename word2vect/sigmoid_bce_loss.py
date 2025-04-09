import torch
from torch import nn


class SigmoidBCELoss(nn.Module):
    def __init__(self, id2word, word2frequency, num_negative_samples=5):
        super(SigmoidBCELoss, self).__init__()
        self.num_negative_samples = num_negative_samples
        self.num_words = len(id2word)
        freq_list = [word2frequency[id2word[i]] for i in range(len(word2frequency))]
        frequencies = torch.tensor(freq_list, dtype=torch.float)

        # 0.75 is a way to make the difference between high freq words vs low freq words
        # a bit less significant in the normalization
        self.distribution = frequencies.pow(0.75)
        self.distribution /= self.distribution.sum()

    def forward(self, input_embedding, output_weights, target_index):
        batch_size = input_embedding.size(0)  # the number of embeddings given as input
        device = input_embedding.device

        # positive context vectors
        pos_embed = output_weights(target_index)  # [batch_size, embedding_dim]
        pos_logits = torch.sum(input_embedding * pos_embed, dim=1)  # [batch_size]
        pos_labels = torch.ones_like(pos_logits)

        # negative sampling
        neg_indices = torch.tensor([
            self.sample(self.num_negative_samples, [target_index[i].item()])
            for i in range(batch_size)
        ], device=device)  # [batch_size, num_neg_samples]

        neg_embed = output_weights(neg_indices)
        input_embed_expanded = input_embedding.unsqueeze(1)
        neg_logits = torch.bmm(neg_embed, input_embed_expanded.transpose(1, 2)).squeeze(2)
        neg_labels = torch.zeros_like(neg_logits)

        # Binary cross entropy loss for positives and negatives
        loss_fn = nn.BCEWithLogitsLoss()

        pos_loss = loss_fn(pos_logits, pos_labels)
        neg_loss = loss_fn(neg_logits, neg_labels)

        return pos_loss + neg_loss

    def sample(self, num_samples, positives=None):
        if positives is None:
            positives = []
        samples = []
        while len(samples) < num_samples:
            w = torch.multinomial(self.distribution, num_samples=1)[0]
            if w.item() not in positives:
                samples.append(w.item())
        return samples
