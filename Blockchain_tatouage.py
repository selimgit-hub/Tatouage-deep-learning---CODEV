import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn.functional as F
from block import block 
import datetime
import hashlib
# Define the transformation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Define the batch size used each time we go through the dataset
batch_size = 32

# Load the dataset
train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)

# Get the dimensions of the first element
first_batch = next(iter(train_loader))
dimensions_premier_element = first_batch[0][0].shape
dimensions_entree = dimensions_premier_element[-1] * dimensions_premier_element[-2]

# Define the model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # Changed input channels to 1 for MNIST
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 64)  # Adjusted to match original architecture
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create an instance of the model
model = SimpleModel()

# Load the weights from the .pth file
model.load_state_dict(torch.load('modele.pth'))

# Set the model to evaluation mode
model.eval()

# Print the weights of the first fully connected layer
fc1_weights = model.fc1.weight.data
fc2_weights = model.fc2.weight.data
fc3_weights = model.fc3.weight.data


# Use the model to make predictions
input_tensor = torch.randn(1, 1, 28, 28)  # Example input tensor with correct dimensions
output = model(input_tensor)
#print("Output tensor:")
#print(output)
combined_weights = torch.cat((fc1_weights.flatten(), fc2_weights.flatten(), fc3_weights.flatten()), dim=0)



###########################################################################
data = combined_weights

############################################################################

block_chain=[block.create_genesis_block()]

print("le block genesis a ete cree !")
print("Hash:%s"% block_chain[-1].hash)

#On va créer par exemple 10 blocks, chaque nouveau block est ajouté au dernier

num_blocks_to_add=10

for i in range(1,num_blocks_to_add+1):
    block_chain.append(block(block_chain[-1].hash,data,datetime.datetime.now()))


    print("Block #%d a ete cree"% i)
    print("\nBlock #%d hash: %s"% (i,block_chain[i].hash))
    print("\nDate pour le block :",block_chain[i].timestamp)
    print("\nBlock #%d a pour data %s  "%(i,block_chain[i].data))

    



