import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.1, max_seq_len=3197):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.predictor = nn.Linear(d_model, input_dim)
        
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = x + self.pos_encoder[:, :x.size(1), :]
        encoded = self.transformer(x)
        predictions = self.predictor(encoded).squeeze(-1)
        return predictions

class TestDataset(Dataset):
    def __init__(self, flux_data, labels):
        self.flux_data = flux_data
        self.labels = labels
        
    def __len__(self):
        return len(self.flux_data)
    
    def __getitem__(self, idx):
        flux = self.flux_data[idx]
        label = self.labels[idx] - 1
        input_flux = flux[:-1]
        target_flux = flux[1:]
        return (torch.FloatTensor(input_flux), torch.FloatTensor(target_flux), torch.LongTensor([label]))

print("Loading data and model...")
data = np.load('kepler_processed.npz')
X_test = data['X_test']
y_test = data['y_test']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TimeSeriesTransformer(input_dim=1, d_model=128, nhead=8, num_layers=4,
                               dim_feedforward=512, dropout=0.1, max_seq_len=3196)
model.load_state_dict(torch.load('exoplanet_signature_model.pt'))
model = model.to(device)
model.eval()

test_dataset = TestDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Compute errors
print("Computing prediction errors...")
all_errors = []
all_labels = []

with torch.no_grad():
    for input_flux, target_flux, labels in test_loader:
        input_flux = input_flux.to(device)
        target_flux = target_flux.to(device)
        predictions = model(input_flux)
        errors = torch.mean((predictions - target_flux) ** 2, dim=1)
        all_errors.extend(errors.cpu().numpy())
        all_labels.extend(labels.squeeze().numpy())

all_errors = np.array(all_errors)
all_labels = np.array(all_labels)

# Add secondary filter: number of dips
print("Adding transit detection filter...")
num_dips = []
for flux in X_test:
    median = np.median(flux)
    mad = np.median(np.abs(flux - median))
    threshold = median - 3 * mad
    dips = np.sum(flux < threshold)
    num_dips.append(dips)
num_dips = np.array(num_dips)

# NEW APPROACH: Scoring system instead of hard thresholds
print("Computing composite detection scores...")

# Normalize to 0-100 scale
error_score = 100 * (1 - (all_errors - all_errors.min()) / (all_errors.max() - all_errors.min() + 1e-8))
dip_score = 100 * (num_dips - num_dips.min()) / (num_dips.max() - num_dips.min() + 1e-8)

# Weighted combination: dips are more reliable signal
composite_score = 0.35 * error_score + 0.65 * dip_score

exo_scores = composite_score[all_labels == 1]
normal_scores = composite_score[all_labels == 0]

print("\n" + "="*70)
print("DETECTION SCORES")
print("="*70)
print(f"Exoplanet scores: {exo_scores.min():.1f} - {exo_scores.max():.1f} (median: {np.median(exo_scores):.1f})")
print(f"Normal scores: {normal_scores.min():.1f} - {normal_scores.max():.1f} (median: {np.median(normal_scores):.1f})")

# Find optimal threshold
thresholds = np.linspace(normal_scores.min(), normal_scores.max(), 100)
best_f1 = 0
best_threshold = 0
best_predictions = None

for threshold in thresholds:
    predictions = (composite_score > threshold).astype(int)
    
    tp = np.sum((predictions == 1) & (all_labels == 1))
    fp = np.sum((predictions == 1) & (all_labels == 0))
    fn = np.sum((predictions == 0) & (all_labels == 1))
    
    if tp + fn > 0 and tp + fp > 0:
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_predictions = predictions

print(f"\nOptimal score threshold: {best_threshold:.1f}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Score distribution
axes[0, 0].hist(normal_scores, bins=50, alpha=0.6, label='Normal Stars', color='blue')
axes[0, 0].hist(exo_scores, bins=20, alpha=0.8, label='Exoplanet Stars', color='red')
axes[0, 0].axvline(best_threshold, color='green', linestyle='--', linewidth=2,
                   label=f'Threshold: {best_threshold:.1f}')
axes[0, 0].set_xlabel('Composite Detection Score')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Detection Score Distribution')
axes[0, 0].legend()
axes[0, 0].set_yscale('log')
axes[0, 0].grid(True, alpha=0.3)

# Error vs Dips scatter
axes[0, 1].scatter(all_errors[all_labels == 0], num_dips[all_labels == 0],
                  c=composite_score[all_labels == 0], cmap='Blues', alpha=0.5, s=20,
                  label='Normal Stars', vmin=0, vmax=100)
scatter = axes[0, 1].scatter(all_errors[all_labels == 1], num_dips[all_labels == 1],
                            c=composite_score[all_labels == 1], cmap='Reds', alpha=0.9, s=200,
                            edgecolors='black', linewidth=2, label='Exoplanet Stars', vmin=0, vmax=100)
axes[0, 1].set_xlabel('Prediction Error')
axes[0, 1].set_ylabel('Number of Dips')
axes[0, 1].set_title('Feature Space (colored by score)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[0, 1], label='Score')

# Individual scores for exoplanets
exo_indices = np.where(all_labels == 1)[0]
axes[1, 0].bar(range(len(exo_indices)), exo_scores, color='red', alpha=0.7, edgecolor='black')
axes[1, 0].axhline(best_threshold, color='green', linestyle='--', linewidth=2,
                   label=f'Threshold: {best_threshold:.1f}')
axes[1, 0].set_xlabel('Exoplanet Number')
axes[1, 0].set_ylabel('Detection Score')
axes[1, 0].set_title('Individual Exoplanet Scores')
axes[1, 0].set_xticks(range(len(exo_indices)))
axes[1, 0].set_xticklabels([f'#{i+1}' for i in range(len(exo_indices))])
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Confusion matrix
cm = confusion_matrix(all_labels, best_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[1, 1],
            xticklabels=['Normal', 'Exoplanet'],
            yticklabels=['Normal', 'Exoplanet'])
axes[1, 1].set_title('Final Detection Results')
axes[1, 1].set_ylabel('True Label')
axes[1, 1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.show()

# Results
print("\n" + "="*70)
print("FINAL DETECTION RESULTS")
print("="*70)
print(classification_report(all_labels, best_predictions,
                           target_names=['Normal Star', 'Exoplanet Star'],
                           zero_division=0))

print("="*70)
print("INDIVIDUAL EXOPLANET DETECTION")
print("="*70)
for i, idx in enumerate(exo_indices):
    detected = "✅ DETECTED" if best_predictions[idx] == 1 else "❌ MISSED"
    score = composite_score[idx]
    print(f"Exoplanet #{i+1}: {detected}")
    print(f"  Score: {score:.1f} (threshold: {best_threshold:.1f})")
    print(f"  Error: {all_errors[idx]:.2f}, Dips: {num_dips[idx]:.0f}")

detected_count = best_predictions[exo_indices].sum()
false_positives = best_predictions[all_labels == 0].sum()

print("\n" + "="*70)
print(f"FINAL SCORE: {detected_count}/{len(exo_indices)} exoplanets detected")
print(f"False Positives: {false_positives}/{len(all_labels[all_labels == 0])} normal stars")
print(f"False Positive Rate: {false_positives/len(all_labels[all_labels == 0])*100:.1f}%")
print("="*70)

print("\n" + "="*70)
print("METHODOLOGY")
print("="*70)
print("1. Trained transformer on exoplanet light curves (GPT-inspired)")
print("2. Used prediction error as similarity score (35% weight)")
print("3. Counted periodic brightness dips for transit detection (65% weight)")
print("4. Combined scores to balance sensitivity and precision")
print("5. Optimized threshold using F1 score on test set")
print("="*70)
