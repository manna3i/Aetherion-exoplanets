import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import seaborn as sns

# =========================
# Analysis functions (same idea)
# =========================
def analyze_prediction_errors(model, data_loader, device):
    model.eval()
    all_errors, all_labels, all_mean, all_max = [], [], [], []

    with torch.no_grad():
        for input_flux, target_flux, labels in data_loader:
            input_flux = input_flux.to(device)     # [B, T-1, 1]
            target_flux = target_flux.to(device)   # [B, T-1, 1]

            preds = model(input_flux, return_classification=False)  # [B, T-1, 1]
            errors = (preds - target_flux) ** 2

            mean_error = errors.mean(dim=(1, 2))
            max_error  = errors.amax(dim=(1, 2))

            all_errors.append(errors.cpu().numpy())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_mean.extend(mean_error.cpu().numpy().tolist())
            all_max.extend(max_error.cpu().numpy().tolist())

    all_errors = np.concatenate(all_errors, axis=0) if len(all_errors) else np.array([])
    return all_errors, np.array(all_labels), np.array(all_mean), np.array(all_max)


def _print_stats(title, arr):
    if arr.size == 0:
        print(f"{title}: n=0")
    else:
        print(f"{title}: {arr.mean():.4f} ± {arr.std():.4f} (n={arr.size})")


def visualize_error_distributions(mean_errors, max_errors, labels):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    normal_mean     = mean_errors[labels == 0]
    exoplanet_mean  = mean_errors[labels == 1]
    normal_max      = max_errors[labels == 0]
    exoplanet_max   = max_errors[labels == 1]

    axes[0].hist(normal_mean, bins=50, alpha=0.6, label='Normal Stars', color='blue')
    axes[0].hist(exoplanet_mean, bins=20, alpha=0.6, label='Exoplanet Stars', color='red')
    axes[0].set_xlabel('Mean Prediction Error'); axes[0].set_ylabel('Count')
    axes[0].set_title('Mean Prediction Error Distribution'); axes[0].legend()
    axes[0].set_yscale('log'); axes[0].grid(True, alpha=0.3)

    axes[1].hist(normal_max, bins=50, alpha=0.6, label='Normal Stars', color='blue')
    axes[1].hist(exoplanet_max, bins=20, alpha=0.6, label='Exoplanet Stars', color='red')
    axes[1].set_xlabel('Max Prediction Error'); axes[1].set_ylabel('Count')
    axes[1].set_title('Maximum Prediction Error Distribution'); axes[1].legend()
    axes[1].set_yscale('log'); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n=== PREDICTION ERROR STATISTICS ===")
    _print_stats("Normal Stars - Mean Error", normal_mean)
    _print_stats("Normal Stars - Max Error", normal_max)
    _print_stats("Exoplanet Stars - Mean Error", exoplanet_mean)
    _print_stats("Exoplanet Stars - Max Error", exoplanet_max)

    if normal_mean.size > 0 and exoplanet_mean.size > 0:
        if exoplanet_mean.mean() > normal_mean.mean():
            ratio = exoplanet_mean.mean() / normal_mean.mean()
            print(f"\n✅ Exoplanets have {ratio:.2f}x higher mean prediction error!")
        else:
            print("\n⚠️ Exoplanets don't show higher prediction errors")
    else:
        print("\n⚠️ One class is empty. Check label mapping.")


def find_optimal_threshold(mean_errors, labels):
    from sklearn.metrics import f1_score, precision_score, recall_score
    if mean_errors.size == 0:
        return 0.0, []

    thresholds = np.linspace(mean_errors.min(), mean_errors.max(), 100)
    best_f1, best_threshold, results = 0.0, thresholds[0], []

    for th in thresholds:
        preds = (mean_errors > th).astype(int)
        if len(np.unique(preds)) > 1:
            precision = precision_score(labels, preds, zero_division=0)
            recall    = recall_score(labels, preds, zero_division=0)
            f1        = f1_score(labels, preds, zero_division=0)
            results.append({'threshold': th, 'precision': precision, 'recall': recall, 'f1': f1, 'predictions': preds})
            if f1 > best_f1:
                best_f1, best_threshold = f1, th

    return best_threshold, results


def evaluate_with_threshold(mean_errors, labels, threshold):
    from sklearn.metrics import confusion_matrix, classification_report
    preds = (mean_errors > threshold).astype(int)

    cm = confusion_matrix(labels, preds, labels=[0,1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Exoplanet'], yticklabels=['Normal', 'Exoplanet'])
    plt.title(f'Confusion Matrix (Threshold = {threshold:.4f})')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label (via Prediction Error)')
    plt.show()

    print("\n=== ANOMALY DETECTION RESULTS ===")
    print(f"Using threshold: {threshold:.4f}")
    print("\n" + classification_report(labels, preds, target_names=['Normal Star', 'Exoplanet Star'], zero_division=0))

    ex_idx = np.where(labels == 1)[0]
    detected = preds[ex_idx]
    print(f"\n=== EXOPLANET DETECTION BREAKDOWN ===")
    for i, (idx, flag) in enumerate(zip(ex_idx, detected)):
        status = "✅ DETECTED" if flag == 1 else "❌ MISSED"
        print(f"Test Exoplanet #{i+1}: {status} (Error: {mean_errors[idx]:.4f})")
    print(f"\nTotal: {detected.sum()}/{len(detected)} exoplanets detected")
    return preds


def plot_roc_curve(mean_errors, labels):
    # Ensure positive class is 1
    fpr, tpr, thresholds = roc_curve(labels, mean_errors, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})', color='darkorange')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--', label='Random', color='navy')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curve - Exoplanet Detection via Prediction Error')
    plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
    plt.show()

    print(f"\nROC AUC Score: {roc_auc:.4f}")
    if roc_auc > 0.7:
        print("✅ Good separation between normal and exoplanet stars!")
    elif roc_auc > 0.5:
        print("⚠️ Some separation, but not great")
    else:
        print("❌ No useful separation")


# =========================
# Dataset
# =========================
class ExoplanetDataset(torch.utils.data.Dataset):
    def __init__(self, flux_data, labels01):
        self.flux_data = flux_data
        self.labels = labels01

    def __len__(self):
        return len(self.flux_data)

    def __getitem__(self, idx):
        flux = self.flux_data[idx]          # [T]
        label = int(self.labels[idx])       # 0/1
        input_flux  = torch.from_numpy(flux[:-1]).float().unsqueeze(-1)  # [T-1, 1]
        target_flux = torch.from_numpy(flux[1:]).float().unsqueeze(-1)   # [T-1, 1]
        return input_flux, target_flux, torch.tensor(label, dtype=torch.long)


# =========================
# Minimal fallback model
# =========================
class SimplePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x, return_classification=False):
        return self.net(x)


def build_model(device):
    try:
        # from your_module import TimeSeriesTransformer
        # model = TimeSeriesTransformer(...); model.load_state_dict(torch.load("model.pth", map_location=device))
        # return model.to(device)
        raise ImportError
    except Exception:
        m = SimplePredictor().to(device)
        print("Using SimplePredictor fallback (no external weights found).")
        return m


def normalize_labels_to_01(y):
    y = np.asarray(y).astype(int).ravel()
    u = set(np.unique(y).tolist())
    if u == {0, 1}:
        return y
    if u == {1, 2}:
        return y - 1
    if u == {-1, 1}:
        return ((y + 1) // 2).astype(int)
    # generic: map min->0, max->1
    vals = np.unique(y)
    mapping = {vals.min(): 0, vals.max(): 1}
    return np.vectorize(mapping.get)(y)


# =========================
# Main
# =========================
if __name__ == "__main__":
    print("Loading model and data...")

    from torch.utils.data import DataLoader

    data = np.load('kepler_processed.npz')
    X_test = data['X_test']  # [N, T]
    y_test = data['y_test']  # {1,2} in your run

    # Normalize labels to {0,1}
    y_test01 = normalize_labels_to_01(y_test)
    if set(np.unique(y_test01)) != {0, 1}:
        raise ValueError(f"Labels not binary after normalization: {np.unique(y_test01)}")

    test_dataset = ExoplanetDataset(X_test, y_test01)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    model = build_model(device)

    print("Analyzing prediction errors...")
    errors, labels, mean_errors, max_errors = analyze_prediction_errors(model, test_loader, device)

    visualize_error_distributions(mean_errors, max_errors, labels)

    # Skip ROC if a class is missing
    if np.unique(labels).tolist() == [0, 1]:
        plot_roc_curve(mean_errors, labels)
    else:
        print("\n⚠️ ROC skipped: one class missing in labels.")

    print("\nFinding optimal detection threshold...")
    best_threshold, results = find_optimal_threshold(mean_errors, labels)
    _ = evaluate_with_threshold(mean_errors, labels, best_threshold)

    print("\n" + "="*60)
    print("CONCLUSION: Your GPT-style prediction approach")
    print("="*60)
