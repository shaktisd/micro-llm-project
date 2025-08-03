import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import os
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import Dict, List, Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader

def visualize_attention(attention_weights: torch.Tensor, tokens: List[str], layer_idx: int, head_idx: int):
    """
    Visualize attention weights for a specific layer and attention head.
    Args:
        attention_weights: Tensor of shape [batch_size, n_heads, seq_len, seq_len]
        tokens: List of input tokens
        layer_idx: Index of the transformer layer
        head_idx: Index of the attention head
    """
    # Get attention weights for the specified head
    attn = attention_weights[0, head_idx].cpu().detach().numpy()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens, annot=True, fmt='.2f', cmap='viridis')
    plt.title(f'Layer {layer_idx}, Head {head_idx} Attention Weights')
    plt.xlabel('Key/Value Tokens')
    plt.ylabel('Query Tokens')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig(f'visualizations/attention_layer{layer_idx}_head{head_idx}.png')
    plt.close()

def print_step_info(step: str, details: str = ""):
    """Print formatted step information during inference"""
    print(f"\n{'='*80}")
    print(f"STEP: {step}")
    if details:
        print(f"Details: {details}")
    print(f"{'='*80}")

# ============================================================================
# 1. TOKENIZER - Convert text to numbers and back
# ============================================================================

class SimpleTokenizer:
    """
    Basic tokenizer that converts text to token IDs and back.
    Uses word-level tokenization with special tokens.
    """
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.BOS_TOKEN = "<BOS>"  # Beginning of sequence
        self.EOS_TOKEN = "<EOS>"  # End of sequence
        
    def build_vocab(self, texts: List[str], min_freq: int = 2):
        """Build vocabulary from training texts"""
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            words = self._tokenize_text(text)
            word_counts.update(words)
        
        # Start with special tokens
        self.word_to_id = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.BOS_TOKEN: 2,
            self.EOS_TOKEN: 3
        }
        
        # Add frequent words to vocabulary
        for word, count in word_counts.items():
            if count >= min_freq:
                self.word_to_id[word] = len(self.word_to_id)
        
        # Create reverse mapping
        self.id_to_word = {id_: word for word, id_ in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)
        
        print(f"Vocabulary size: {self.vocab_size}")
        
    def _tokenize_text(self, text: str) -> List[str]:
        """Simple word tokenization"""
        # Convert to lowercase and split on whitespace and punctuation
        text = text.lower()
        words = re.findall(r'\b\w+\b|[^\w\s]', text)
        return words
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert text to token IDs"""
        words = self._tokenize_text(text)
        
        # Convert words to IDs
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.word_to_id[self.BOS_TOKEN])
            
        for word in words:
            token_id = self.word_to_id.get(word, self.word_to_id[self.UNK_TOKEN])
            token_ids.append(token_id)
            
        if add_special_tokens:
            token_ids.append(self.word_to_id[self.EOS_TOKEN])
            
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text"""
        words = []
        for token_id in token_ids:
            if token_id in self.id_to_word:
                word = self.id_to_word[token_id]
                if word not in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]:
                    words.append(word)
        return ' '.join(words)

# ============================================================================
# 2. DATASET - Prepare training data
# ============================================================================

class TextDataset(Dataset):
    """Dataset for language modeling"""
    def __init__(self, texts: List[str], tokenizer: SimpleTokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences = []
        
        # Convert texts to sequences
        for text in texts:
            token_ids = tokenizer.encode(text)
            
            # Split long sequences into chunks
            for i in range(0, len(token_ids) - 1, max_length - 1):
                seq = token_ids[i:i + max_length]
                if len(seq) > 1:  # Need at least 2 tokens (input + target)
                    self.sequences.append(seq)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Pad if necessary
        if len(seq) < self.max_length:
            seq = seq + [self.tokenizer.word_to_id[self.tokenizer.PAD_TOKEN]] * (self.max_length - len(seq))
        
        # Input is all tokens except last, target is all tokens except first
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        target_ids = torch.tensor(seq[1:], dtype=torch.long)
        
        return input_ids, target_ids

# ============================================================================
# 3. TRANSFORMER ARCHITECTURE - The core model
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism - the core of transformers.
    Allows the model to attend to different parts of the sequence simultaneously.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for queries, keys, and values
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.last_attention_weights = None  # Store for visualization
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, debug: bool = False):
        batch_size, seq_len, _ = x.shape
        
        if debug:
            print_step_info("Multi-Head Attention", 
                          f"Input shape: {x.shape}, Heads: {self.n_heads}, Head dim: {self.d_k}")
        
        # Linear projections and reshape for multi-head attention
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        if debug:
            print(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
            print("\nAttention computation steps:")
            print("1. Project input into Query, Key, Value vectors for each attention head")
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if debug:
            print("2. Compute attention scores: Q × K^T / sqrt(d_k)")
            print(f"Raw attention scores shape: {scores.shape}")
        
        # Apply causal mask (for autoregressive generation)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            if debug:
                print("3. Apply causal mask to prevent attending to future tokens")
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        if debug:
            print("4. Apply softmax to get attention probabilities")
            print(f"Attention weights shape: {attn_weights.shape}")
        
        # Store attention weights for visualization
        self.last_attention_weights = attn_weights.detach()
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        if debug:
            print("5. Apply attention weights to Values")
            print(f"Context vectors shape: {context.shape}")
        
        # Concatenate heads and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        
        if debug:
            print("6. Concatenate all heads and project to output dimension")
            print(f"Final output shape: {output.shape}")
        
        return output

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    Processes each position independently with two linear transformations.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """
    Single transformer block with self-attention and feed-forward layers.
    Uses residual connections and layer normalization.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, debug: bool = False):
        if debug:
            print_step_info("Transformer Block", f"Input shape: {x.shape}")
            print("1. Applying Layer Normalization before Self-Attention")
        
        # Self-attention with residual connection
        normalized_x = self.norm1(x)
        attn_output = self.attention(normalized_x, mask, debug=debug)
        x = x + self.dropout(attn_output)
        
        if debug:
            print("2. Add residual connection from input")
            print(f"Shape after attention: {x.shape}")
            print("\n3. Applying Layer Normalization before Feed-Forward")
        
        # Feed-forward with residual connection
        normalized_x = self.norm2(x)
        ff_output = self.feed_forward(normalized_x)
        x = x + self.dropout(ff_output)
        
        if debug:
            print("4. Add residual connection from attention output")
            print(f"Final output shape: {x.shape}")
        
        return x

class PositionalEncoding(nn.Module):
    """
    Adds positional information to token embeddings.
    Uses sinusoidal encoding to help the model understand token positions.
    """
    def __init__(self, d_model: int, max_length: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class MicroLLM(nn.Module):
    """
    Complete micro language model using transformer architecture.
    Similar to GPT but much smaller for demonstration purposes.
    """
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int = 256,      # Embedding dimension
        n_heads: int = 8,        # Number of attention heads
        n_layers: int = 6,       # Number of transformer blocks
        d_ff: int = 1024,        # Feed-forward dimension
        max_length: int = 128,   # Maximum sequence length
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_length = max_length
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_length)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layer
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_causal_mask(self, seq_len: int, device: torch.device):
        """Create causal mask to prevent attention to future tokens"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    
    def forward(self, input_ids: torch.Tensor):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len, device)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, causal_mask)
        
        # Final layer norm and output projection
        x = self.layer_norm(x)
        logits = self.output_projection(x)
        
        return logits

# ============================================================================
# 4. DATA LOADING - Download and prepare training data
# ============================================================================

def download_tiny_shakespeare():
    """
    Download the Tiny Shakespeare dataset - a small, publicly available text corpus
    perfect for demonstrating language model training.
    """
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    training_data = r"data\short_stories.txt"
    if not os.path.exists(training_data):
        print("Data not found")
        #response = requests.get(url)
        #with open("tiny_shakespeare.txt", "w", encoding="utf-8") as f:
        #    f.write(response.text)
        #print("Dataset downloaded!")
    else:
        print("Dataset already exists!")
    
    with open(training_data, "r", encoding="utf-8") as f:
        text = f.read()
    
    return text

def prepare_training_data(text: str, train_split: float = 0.9):
    """Split text into training and validation sets"""
    # Split into sentences for better training examples
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
    
    # Split into train/val
    split_idx = int(len(sentences) * train_split)
    train_texts = sentences[:split_idx]
    val_texts = sentences[split_idx:]
    
    print(f"Training sentences: {len(train_texts)}")
    print(f"Validation sentences: {len(val_texts)}")
    
    return train_texts, val_texts

# ============================================================================
# 5. TRAINING LOOP - Pre-training the model
# ============================================================================

def train_model(model, train_loader, val_loader, tokenizer, epochs=10, lr=1e-4):
    """
    Pre-training loop using next-token prediction.
    This is the same objective used by GPT models.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word_to_id[tokenizer.PAD_TOKEN])
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            
            # Forward pass
            logits = model(input_ids)
            
            # Calculate loss (predict next token)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (important for stable training)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_train_loss = total_loss / num_batches
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids, target_ids = input_ids.to(device), target_ids.to(device)
                logits = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Generate sample text to monitor progress
        if (epoch + 1) % 5 == 0:
            sample_text = generate_text(model, tokenizer, "The king", max_length=20)
            print(f"Sample generation: {sample_text}")
            print("-" * 50)

# ============================================================================
# 6. INFERENCE - Text generation after training
# ============================================================================

def generate_text(
    model, 
    tokenizer, 
    prompt: str, 
    max_length: int = 50, 
    temperature: float = 1.0,
    top_k: int = 50,
    visualize: bool = True,
    debug: bool = True
):
    """
    Generate text using the trained model with visualization and debugging.
    Implements several sophisticated sampling strategies.
    """
    device = next(model.parameters()).device
    model.eval()
    
    if debug:
        print_step_info("Text Generation", f"Prompt: '{prompt}'")
        print("Model Configuration:")
        print(f"Temperature: {temperature}")
        print(f"Top-k: {top_k}")
        print(f"Max length: {max_length}")
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    if debug:
        print("\nTokenized prompt:")
        tokens = [tokenizer.id_to_word[id_] for id_ in input_ids[0].tolist()]
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {input_ids[0].tolist()}")
    
    generated_ids = input_ids.clone()
    generated_tokens = []
    
    with torch.no_grad():
        for step in range(max_length):
            if debug:
                print(f"\nGeneration Step {step + 1}")
                print("-" * 40)
            
            # Get model predictions with debugging
            logits = model(generated_ids)
            
            # Get logits for the last token
            last_token_logits = logits[0, -1, :] / temperature
            
            if debug:
                print("\nToken Selection:")
                print(f"1. Applied temperature {temperature} to logits")
            
            # Top-k sampling: only consider top k most likely tokens
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(last_token_logits, top_k)
                last_token_logits = torch.full_like(last_token_logits, -float('inf'))
                last_token_logits[top_k_indices] = top_k_logits
                
                if debug:
                    print(f"2. Selected top {top_k} tokens")
                    top_k_words = [tokenizer.id_to_word[idx.item()] for idx in top_k_indices[:5]]
                    top_k_probs = F.softmax(top_k_logits[:5], dim=0)
                    print("Top 5 candidates:")
                    for word, prob in zip(top_k_words, top_k_probs):
                        print(f"   {word}: {prob:.3f}")
            
            # Sample from the probability distribution
            probs = F.softmax(last_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            if debug:
                print(f"3. Sampled token: '{tokenizer.id_to_word[next_token.item()]}'")
            
            # Visualize attention patterns
            if visualize and step % 2 == 0:  # Visualize every other step to reduce clutter
                for layer_idx in range(len(model.transformer_blocks)):
                    block = model.transformer_blocks[layer_idx]
                    if hasattr(block.attention, 'last_attention_weights'):
                        current_tokens = [tokenizer.id_to_word[id_] for id_ in generated_ids[0].tolist()]
                        for head in range(block.attention.n_heads):
                            visualize_attention(
                                block.attention.last_attention_weights,
                                current_tokens,
                                layer_idx,
                                head
                            )
            
            # Stop if we generate end token
            if next_token.item() == tokenizer.word_to_id[tokenizer.EOS_TOKEN]:
                if debug:
                    print("\nGenerated EOS token, stopping generation")
                break
            
            # Append the new token
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
            generated_tokens.append(tokenizer.id_to_word[next_token.item()])
            
            if debug:
                print(f"Current text: {prompt} {''.join(generated_tokens)}")
            
            # Prevent infinite generation
            if generated_ids.size(1) > model.max_length:
                if debug:
                    print("\nReached maximum length, stopping generation")
                break
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())
    return generated_text

def interactive_chat(model, tokenizer):
    """Simple chat interface for the trained model"""
    print("\n" + "="*50)
    print("MICRO LLM CHAT INTERFACE")
    print("Type 'quit' to exit")
    print("="*50)
    
    while True:
        prompt = input("\nYou: ").strip()
        if prompt.lower() in ['quit', 'exit']:
            break
        
        if prompt:
            response = generate_text(
                model, tokenizer, prompt, 
                max_length=30, temperature=0.8, top_k=40
            )
            print(f"Model: {response}")

# ============================================================================
# 7. MAIN EXECUTION - Complete training pipeline
# ============================================================================

def main():
    """
    Complete pipeline demonstrating:
    1. Data preparation
    2. Model architecture
    3. Pre-training
    4. Inference and generation
    """
    print("MICRO LANGUAGE MODEL DEMONSTRATION")
    print("=" * 50)
    
    # Step 1: Download and prepare data
    print("\n1. PREPARING TRAINING DATA")
    text = download_tiny_shakespeare()
    train_texts, val_texts = prepare_training_data(text)
    
    # Step 2: Build tokenizer
    print("\n2. BUILDING TOKENIZER")
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(train_texts + val_texts, min_freq=3)
    
    # Step 3: Create datasets
    print("\n3. CREATING DATASETS")
    max_seq_length = 64  # Shorter for faster training
    train_dataset = TextDataset(train_texts, tokenizer, max_seq_length)
    val_dataset = TextDataset(val_texts, tokenizer, max_seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Step 4: Initialize model
    print("\n4. INITIALIZING MODEL")
    model = MicroLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,    # Smaller for demo
        n_heads=4,      # Fewer heads
        n_layers=3,     # Fewer layers
        d_ff=512,       # Smaller feed-forward
        max_length=max_seq_length,
        dropout=0.1
    )
    
    # Step 5: Train the model
    print("\n5. TRAINING MODEL")
    print("This demonstrates the pre-training phase where the model learns language patterns")
    train_model(model, train_loader, val_loader, tokenizer, epochs=15, lr=3e-4)
    
    # Step 6: Demonstrate inference
    print("\n6. INFERENCE DEMONSTRATION")
    print("Testing various generation strategies:")
    
    test_prompts = [
        "To be or not to be",
        "Romeo and Juliet",
        "The king said",
        "In fair Verona"
    ]
    
    for prompt in test_prompts:
        generated = generate_text(
            model, tokenizer, prompt, 
            max_length=25, temperature=0.8, top_k=30
        )
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated}'")
        print()
    
    # Step 7: Interactive chat
    print("\n7. INTERACTIVE CHAT")
    interactive_chat(model, tokenizer)
    
    # Save model and tokenizer
    print("\nSaving model and tokenizer...")
    torch.save(model.state_dict(), "micro_llm_model.pt")
    import pickle
    with open("micro_llm_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print("Model and tokenizer saved!")
    
    return model, tokenizer

# ============================================================================
# 8. ADVANCED FEATURES DEMONSTRATION
# ============================================================================

def demonstrate_advanced_features(model, tokenizer):
    """
    Demonstrate advanced LLM features like:
    - Different sampling strategies
    - Attention visualization
    - Perplexity calculation
    """
    print("\nADVANCED FEATURES DEMONSTRATION")
    print("=" * 40)
    
    prompt = "The fair Juliet"
    
    print(f"Prompt: '{prompt}'")
    print("\nDifferent sampling strategies:")
    
    # Greedy decoding (temperature = 0)
    greedy = generate_text(model, tokenizer, prompt, max_length=15, temperature=0.1)
    print(f"Greedy (temp=0.1): {greedy}")
    
    # High temperature (more random)
    random_gen = generate_text(model, tokenizer, prompt, max_length=15, temperature=1.5)
    print(f"Random (temp=1.5): {random_gen}")
    
    # Balanced sampling
    balanced = generate_text(model, tokenizer, prompt, max_length=15, temperature=0.8)
    print(f"Balanced (temp=0.8): {balanced}")

def calculate_perplexity(model, tokenizer, text_samples):
    """Calculate perplexity - a measure of how well the model predicts text"""
    device = next(model.parameters()).device
    model.eval()
    
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word_to_id[tokenizer.PAD_TOKEN])
    
    with torch.no_grad():
        for text in text_samples[:10]:  # Sample a few texts
            input_ids = tokenizer.encode(text)
            if len(input_ids) < 2:
                continue
                
            input_tensor = torch.tensor([input_ids[:-1]], dtype=torch.long).to(device)
            target_tensor = torch.tensor([input_ids[1:]], dtype=torch.long).to(device)
            
            if input_tensor.size(1) > model.max_length:
                input_tensor = input_tensor[:, :model.max_length]
                target_tensor = target_tensor[:, :model.max_length]
            
            logits = model(input_tensor)
            loss = criterion(logits.view(-1, logits.size(-1)), target_tensor.view(-1))
            
            total_loss += loss.item() * target_tensor.numel()
            total_tokens += target_tensor.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

# =========================================================================
# 9. LOAD AND INFER FROM SAVED MODEL
# =========================================================================
def load_model_and_infer(prompt: str, model_path: str = "micro_llm_model.pt", tokenizer_path: str = "micro_llm_tokenizer.pkl"):
    """
    Load a saved model and tokenizer, then run inference on a prompt.
    """
    import pickle
    # Load tokenizer
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    # Create model with correct vocab size and config
    model = MicroLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=3,
        d_ff=512,
        max_length=64,
        dropout=0.1
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    model.eval()
    
    interactive_chat(model, tokenizer)

# ============================================================================
# RUN THE COMPLETE DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    TRAIN = False
    if TRAIN:
        # Run the complete pipeline
        model, tokenizer = main()
        
        # Demonstrate advanced features
        demonstrate_advanced_features(model, tokenizer)
        
        print(f"\n🎉 DEMONSTRATION COMPLETE!")
        print(f"You've just seen a complete micro language model with:")
        print(f"• Transformer architecture with self-attention")
        print(f"• Pre-training on next-token prediction")
        print(f"• Sophisticated text generation")
        print(f"• All major LLM components in ~400 lines of code!")
    else:    
        # Example: Run inference from saved model
        # Uncomment below to test loading and inference
        load_model_and_infer("Why are you in such")
