import torch
import torch.nn as nn
import torch.optim as optim
from agent.networks import CNN


class BCAgent:

    def __init__(self, history_length=1, n_classes=5, lr=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define network, loss function, optimizer
        self.net = CNN(history_length=history_length, n_classes=n_classes).to(
            self.device
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def update(self, X_batch, y_batch):
        # Transform input to tensors
        X_tensor = torch.FloatTensor(X_batch).to(self.device)
        y_tensor = torch.LongTensor(y_batch).to(self.device)

        # Forward + backward + optimize
        self.optimizer.zero_grad()
        outputs = self.net(X_tensor)
        loss = self.loss_fn(outputs, y_tensor)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, X):
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.net.eval()
        with torch.no_grad():
            outputs = self.net(X_tensor)
        self.net.train()
        return outputs

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name, map_location=self.device))

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)
