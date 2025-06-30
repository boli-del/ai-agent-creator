import torch
import torch.nn as nn
import torch.nn.functional as F

class SlotClassifier(nn.Module):

    def __init__(
        self,
        input_dim: int,
        num_slots: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_layers: int = 2
    ):
        """
        Initialize the slot classifier.
        input_dim: Dimension of the input features (usually dimension_of_model or d_model from transformer)
        num_slots: Number of different slot types to classify
        hidden_dim: Dimension of hidden layers in the MLP
        dropout: Dropout probability for regularization
        num_layers: Number of hidden layers in the MLP
        """
        super().__init__()
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Add final classification layer
        layers.append(nn.Linear(prev_dim, num_slots))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the slot classifier.
        x: Input tensor of shape [batch_size, input_dim]
        Usually the [CLS] token representation from the transformer
        """
        logits = self.mlp(x)
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predictions from the classifier.
        x: Input tensor of shape [batch_size, input_dim]
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)
    
    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability distribution over slots.
        x: Input tensor of shape [batch_size, input_dim]
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1) 