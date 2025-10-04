import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Simple Transformer for Time Series Prediction
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.1, max_seq_len=3197):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # For next-value prediction
        self.predictor = nn.Linear(d_model, input_dim)
        
        # For classification (optional)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        
    def forward(self, x, return_classification=False):
        # x shape: (batch, seq_len)
        x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        
        # Embed and add positional encoding
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = x + self.pos_encoder[:, :x.size(1), :]
        
        # Transform
        encoded = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Next-value prediction
        predictions = self.predictor(encoded).squeeze(-1)  # (batch, seq_len)
        
        if return_classification:
            # Use mean of encoded sequence for classification
            pooled = encoded.mean(dim=1)  # (batch, d_model)
            classification = self.classifier(pooled)  # (batch, 2)
            return predictions, classification
        
        return predictions

class ExoplanetDataset(Dataset):
    def __init__(self, flux_data, labels, predict_horizon=1):
        self.flux_data = flux_data
        self.labels = labels
        self.predict_horizon = predict_horizon
        
    def __len__(self):
        return len(self.flux_data)
    
    def __getitem__(self, idx):
        flux = self.flux_data[idx]
        label = self.labels[idx] - 1  # Convert 1,2 to 0,1
        
        # For next-value prediction: shift by predict_horizon
        input_flux = flux[:-self.predict_horizon]
        target_flux = flux[self.predict_horizon:]
        
        return (
            torch.FloatTensor(input_flux),
            torch.FloatTensor(target_flux),
            torch.LongTensor([label])
        )

def train_model(model, train_loader, val_loader, device, num_epochs=20, lr=0.001):
    """Train the transformer model"""
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Two loss functions
    prediction_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 10.0]).to(device)  # Weight rare class heavily
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, 
    )
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (input_flux, target_flux, labels) in enumerate(train_loader):
            input_flux = input_flux.to(device)
            target_flux = target_flux.to(device)
            labels = labels.to(device).squeeze()
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions, classification = model(input_flux, return_classification=True)
            
            # Calculate losses
            pred_loss = prediction_criterion(predictions, target_flux)
            class_loss = classification_criterion(classification, labels)
            
            # Combined loss (prediction + classification)
            loss = pred_loss + 0.5 * class_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(classification, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for input_flux, target_flux, labels in val_loader:
                input_flux = input_flux.to(device)
                target_flux = target_flux.to(device)
                labels = labels.to(device).squeeze()
                
                predictions, classification = model(input_flux, return_classification=True)
                
                pred_loss = prediction_criterion(predictions, target_flux)
                class_loss = classification_criterion(classification, labels)
                loss = pred_loss + 0.5 * class_loss
                
                val_loss += loss.item()
                
                _, predicted = torch.max(classification, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_exoplanet_model.pt')
            print(f'✅ Saved best model with val_loss: {val_loss:.4f}')
        
        scheduler.step(val_loss)
    
    return history

def evaluate_model(model, test_loader, device):
    """Evaluate the model and generate predictions"""
    
    model.eval()
    all_preds = []
    all_labels = []
    all_pred_errors = []
    
    with torch.no_grad():
        for input_flux, target_flux, labels in test_loader:
            input_flux = input_flux.to(device)
            target_flux = target_flux.to(device)
            
            predictions, classification = model(input_flux, return_classification=True)
            
            # Prediction error (anomaly score)
            pred_error = torch.mean((predictions - target_flux) ** 2, dim=1)
            
            _, predicted = torch.max(classification, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.squeeze().numpy())
            all_pred_errors.extend(pred_error.cpu().numpy())
    
    # Print classification report
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(all_labels, all_preds, 
                                target_names=['Normal Star', 'Exoplanet Star']))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Exoplanet'],
                yticklabels=['Normal', 'Exoplanet'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return all_preds, all_labels, all_pred_errors

def plot_training_history(history):
    """Plot training metrics"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load preprocessed data
    print("Loading preprocessed data...")
    data = np.load('kepler_processed.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"Loaded: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
    
    # Create datasets
    train_dataset = ExoplanetDataset(X_train, y_train)
    test_dataset = ExoplanetDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = TimeSeriesTransformer(
        input_dim=1,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_len=3196  # One less due to prediction shift
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n🚀 Starting training...")
    history = train_model(model, train_loader, test_loader, device, 
                         num_epochs=15, lr=0.0005)
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model and evaluate
    print("\n📊 Evaluating best model...")
    model.load_state_dict(torch.load('best_exoplanet_model.pt'))
    preds, labels, errors = evaluate_model(model, test_loader, device)
    
    print("\n✅ Training complete!")
    print("Model saved as 'best_exoplanet_model.pt'")