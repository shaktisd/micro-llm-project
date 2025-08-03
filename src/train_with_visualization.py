#!/usr/bin/env python3
"""
Enhanced training script with comprehensive visualization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from micro_llm import *
from training_utils import TrainingVisualizer, visualize_attention
import matplotlib.pyplot as plt
import time

def enhanced_train_model(model, train_loader, val_loader, tokenizer, epochs=20):
    """Enhanced training with visualization"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize visualizer
    visualizer = TrainingVisualizer()
    
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word_to_id[tokenizer.PAD_TOKEN])
    
    print(f"🚀 Enhanced Training Started!")
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        total_loss = 0
        total_grad_norm = 0
        num_batches = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            
            # Forward pass
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Calculate gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            total_grad_norm += grad_norm.item()
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Progress indicator
            if batch_idx % 20 == 0:
                progress = (batch_idx / len(train_loader)) * 100
                print(f"\rEpoch {epoch+1}/{epochs} - Batch {batch_idx}/{len(train_loader)} "
                      f"({progress:.1f}%) - Loss: {loss.item():.4f}", end="")
        
        scheduler.step()
        avg_train_loss = total_loss / num_batches
        avg_grad_norm = total_grad_norm / num_batches
        
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
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - start_time
        
        # Log metrics
        visualizer.log_epoch(epoch + 1, avg_train_loss, avg_val_loss, current_lr, avg_grad_norm)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1}/{epochs} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss: {avg_val_loss:.4f}")
        print(f"   Perplexity: {np.exp(avg_val_loss):.2f}")
        print(f"   Learning Rate: {current_lr:.2e}")
        print(f"   Grad Norm: {avg_grad_norm:.4f}")
        print(f"   Time: {epoch_time:.2f}s")
        
        # Generate sample text every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"\n✨ Sample Generation:")
            for prompt in ["To be", "The king", "Love is"]:
                sample = generate_text(model, tokenizer, prompt, max_length=15, temperature=0.8)
                print(f"   '{prompt}' → '{sample}'")
        
        print("-" * 60)
    
    # Final visualization
    print("\n🎨 Creating training visualizations...")
    visualizer.plot_training_curves()
    visualizer.save_history()
    
    # Attention visualization
    print("🔍 Generating attention visualization...")
    visualize_attention(model, tokenizer, "To be or not to be", layer=0, head=0)
    
    return model, visualizer



def main_enhanced():
    """Enhanced main function with better organization"""
    print("🧠 MICRO LANGUAGE MODEL - ENHANCED VERSION")
    print("=" * 50)
    
    # Step 1: Data preparation
    print("\n📚 STEP 1: DATA PREPARATION")
    try:
        text = download_tiny_shakespeare()
        train_texts, val_texts = prepare_training_data(text)
        print("✅ Data loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Step 2: Tokenizer
    print("\n🔤 STEP 2: TOKENIZER SETUP")
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(train_texts + val_texts, min_freq=2)
    print("✅ Tokenizer built successfully!")
    
    # Step 3: Datasets
    print("\n📊 STEP 3: DATASET CREATION")
    max_seq_length = 64
    train_dataset = TextDataset(train_texts, tokenizer, max_seq_length)
    val_dataset = TextDataset(val_texts, tokenizer, max_seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    print("✅ Datasets created successfully!")
    
    # Step 4: Model
    print("\n🏗️ STEP 4: MODEL INITIALIZATION")
    model = MicroLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_length=max_seq_length,
        dropout=0.1
    )
    print("✅ Model initialized successfully!")
    
    # Step 5: Enhanced training
    print("\n🎯 STEP 5: ENHANCED TRAINING")
    model, visualizer = enhanced_train_model(
        model, train_loader, val_loader, tokenizer, epochs=20
    )
    
    # Step 6: Save model
    print("\n💾 STEP 6: SAVING MODEL")
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'training_history': visualizer.history
    }, 'models/micro_llm_trained.pth')
    print("✅ Model saved successfully!")
    
    # Step 7: Interactive demo
    print("\n🎮 STEP 7: INTERACTIVE DEMO")
    interactive_chat(model, tokenizer)
    
if __name__ == "__main__":
    # Run enhanced training
    main_enhanced()