import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict
import json
import os

class TrainingVisualizer:
    """Enhanced training visualization utilities"""
    
    def __init__(self, save_dir: str = "training_logs"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'grad_norms': []
        }
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                  lr: float, grad_norm: float):
        """Log training metrics for an epoch"""
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['learning_rates'].append(lr)
        self.history['grad_norms'].append(grad_norm)
    
    def plot_training_curves(self):
        """Create comprehensive training plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        ax1.plot(self.history['epochs'], self.history['train_loss'], 
                label='Training Loss', color='#FF6B6B', linewidth=2)
        ax1.plot(self.history['epochs'], self.history['val_loss'], 
                label='Validation Loss', color='#4ECDC4', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning rate
        ax2.plot(self.history['epochs'], self.history['learning_rates'], 
                color='#FFD700', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Gradient norms
        ax3.plot(self.history['epochs'], self.history['grad_norms'], 
                color='#9B59B6', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Gradient Norm')
        ax3.set_title('Gradient Norms')
        ax3.grid(True, alpha=0.3)
        
        # Perplexity
        perplexity = [np.exp(loss) for loss in self.history['val_loss']]
        ax4.plot(self.history['epochs'], perplexity, 
                color='#E74C3C', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Perplexity')
        ax4.set_title('Validation Perplexity')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_history(self):
        """Save training history to JSON"""
        with open(f'{self.save_dir}/training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

def visualize_attention(model, tokenizer, text: str, layer: int = 0, head: int = 0):
    """Visualize attention patterns for a given text"""
    model.eval()
    
    # Tokenize input
    tokens = tokenizer.encode(text, add_special_tokens=False)
    input_ids = torch.tensor([tokens])
    
    # Get attention weights (this would need to be implemented in the model)
    with torch.no_grad():
        # This is a simplified version - you'd need to modify the model
        # to return attention weights
        outputs = model(input_ids)
    
    # For demonstration, create a random attention pattern
    seq_len = len(tokens)
    attention_weights = torch.rand(seq_len, seq_len)
    attention_weights = torch.tril(attention_weights)  # Causal mask
    
    # Normalize
    for i in range(seq_len):
        attention_weights[i, :i+1] = torch.softmax(attention_weights[i, :i+1], dim=0)
    
    # Plot
    plt.figure(figsize=(10, 8))
    word_tokens = [tokenizer.id_to_word.get(token_id, '<UNK>') for token_id in tokens]
    
    sns.heatmap(attention_weights.numpy(), 
                xticklabels=word_tokens,
                yticklabels=word_tokens,
                cmap='YlOrRd',
                cbar_kws={'label': 'Attention Weight'})
    
    plt.title(f'Attention Pattern - Layer {layer}, Head {head}')
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()