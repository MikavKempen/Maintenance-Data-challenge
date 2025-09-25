"""
    1CM290 Maintenance Optimziation and Engineering (Lecturer: J. Lee)
    Assignment: Data Cahllenges
    Challenge: Detection of faults in gears.
    This is a template for the assignment.
    You may fill the parts <YOUR CODE HERE> or add new parts as you need.
    You may change as you need.
    Please make concise and comprehensive comments.
"""

import numpy as np
from matplotlib import pyplot as plt
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Read pre-processed data
data_train_label = np.load('data_gear_train_label.npy')
data_train_feature = np.load('data_gear_train_feature.npy')
data_valid_label = np.load('data_gear_valid_label.npy')
data_valid_feature = np.load('data_gear_valid_feature.npy')
data_test_label = np.load('data_gear_test_label.npy')
data_test_feature = np.load('data_gear_test_feature.npy')
n_feature = data_train_feature.shape[1]  # YOUR CODE HERE


# Prepare dataset
class MyDataset(Dataset):
    def __init__(self, feature, label):
        self.x = T.tensor(feature, dtype=T.float32)
        self.y = T.tensor(label, dtype=T.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


dataset_train = MyDataset(data_train_feature, data_train_label)
dataset_valid = MyDataset(data_valid_feature, data_valid_label)
dataset_test = MyDataset(data_test_feature, data_test_label)
train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
valid_loader = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=True)


# Task (b) Train your ANN model
class Net(nn.Module):
    def __init__(self, n_feature):
        super().__init__()
        self.chkpt_dir = '/tmp_gear'

        # YOUR CODE HERE

        # If you have GPU for CUDA, you can use this.
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        # YOUR CODE HERE

        return x

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_dir)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_dir))


# YOUR CODE HERE
net = Net(n_feature)


# Define criterion
# YOUR CODE HERE
criterion = None  # YOUR CODE HERE

# Define optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)

num_epoch = 0  # YOUR CODE HERE
train_loss, train_epoch = [], []

for epoch in range(num_epoch):
    net.train()
    for batch_x, batch_y in train_loader:
        # YOUR CODE HERE
        pass

    # YOUR CODE HERE

# YOUR CODE HERE

# Task (c) Evaluate the final performance
with T.no_grad():
    net.eval()
    for batch_x, batch_y in test_loader:
        # YOUR CODE HERE
        pass

# YOUR CODE HERE
