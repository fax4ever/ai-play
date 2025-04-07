import torch, os
from tqdm.auto import tqdm
from sigmoid_bce_loss import SigmoidBCELoss

class Trainer:
    def __init__(self, model, optimizer, device):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.model.train()
        self.model.to(device)

    def train(self, train_dataset, output_folder, epochs=1):
        train_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            for step, sample in tqdm(enumerate(train_dataset), desc="Batch", leave=False):
                inputs = sample['inputs'].to(self.device)
                targets = sample['targets'].to(self.device)

                if isinstance(self.model.loss_function, SigmoidBCELoss):
                    loss = self.model(inputs, targets)
                else:
                    output_distribution = self.model(inputs)
                    loss = self.model.loss_function(output_distribution, targets)

                loss.backward()
                # updates the parameters
                # by applying the gradients to update all the parameters (weights and biases) of the model
                # as computed using backpropagation during the call to loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches
            print(f'Epoch: {epoch} avg loss = {avg_epoch_loss:.4f}')

            train_loss += avg_epoch_loss
            torch.save(self.model.state_dict(), os.path.join(output_folder, f'state_{epoch}.pt'))  # save the model state
        return train_loss / epochs