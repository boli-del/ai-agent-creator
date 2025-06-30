import torch
import torch.nn as nn
#simple test for the linear function to determine it's properties
m = nn.Linear(20, 30)
input = torch.randn(128, 20)
output = m(input)
print(output.size())
print(output)