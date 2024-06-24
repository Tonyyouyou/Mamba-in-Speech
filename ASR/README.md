# Overview
The following files are used for implementing ASR tasks, specifically the relevant ESPnet files.

## Outer_bimamba.py
This file contains ExtBimamba, which we proposed in our paper. should be placed in mamba_ssm.modules(in your enviroment package)

## conformer_encoder_mamba.py
should be placed in espnet/espnet2/asr/encoder

## encoder_layer_mamba.py
shoule be placed in espnet/espnet/nets/pytorch_backend/conformer

## encoder_mamba.py
shoule be placed in espnet/espnet/nets/pytorch_backend/conformer

## asr.py
you also need to change a liite bit for asr.py file, which is in espnet/espnet2/tasks

## example_yaml.yaml
This file is an example file, showing how it should look after you place the above files in the correct location.

# Pytorch version of ConBiMamba
## Usage

```python
# Import necessary libraries
import torch
import torch.nn as nn
from ConExBiMamba.model import ConBiMamba

# Set batch size, sequence length, and dimension
batch_size, sequence_length, dim = 3, 12345, 80

# Check for CUDA availability and set device
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

# Define the criterion (CTCLoss)
criterion = nn.CTCLoss().to(device)

# Generate random inputs
inputs = torch.rand(batch_size, sequence_length, dim).to(device)

# Define input lengths
input_lengths = torch.LongTensor([12345, 12300, 12000])

# Define targets
targets = torch.LongTensor([
    [1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
    [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
    [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]
]).to(device)

# Define target lengths
target_lengths = torch.LongTensor([9, 8, 7])

# Initialize the model
model = Conformer(
    num_classes=10, 
    input_dim=dim, 
    encoder_dim=32, 
    num_encoder_layers=3
).to(device)

# Forward propagate
outputs, output_lengths = model(inputs, input_lengths)

# Calculate CTC Loss
loss = criterion(outputs.transpose(0, 1), targets, output_lengths, target_lengths)
```
We Thanks to Author of https://github.com/sooftware/conformer