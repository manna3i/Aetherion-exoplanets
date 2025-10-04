# **見ろ (Look!)**

The Kepler Mission, a NASA Discovery mission launched on March 6, 2009, was the first space mission dedicated to the search for Earth-sized and smaller planets in the habitable zone of other stars in our neighbourhood of the galaxy. Kepler was a special-purpose spacecraft that precisely measured the light variations from thousands of distant stars, looking for planetary transits. When a planet passes in front of its parent star, as seen from our solar system, it blocks a small fraction of the light from that star; this is known as a transit. Searching for transits of distant Earths is like looking for the drop in brightness when a moth flies across a searchlight. Measuring repeated transits, all with a regular period, duration and change in brightness, provides a method for discovering planets and measuring the duration of their orbits—planets the size of Earth and smaller in the habitable zone around other stars similar to our Sun. Kepler continuously monitored over 100,000 stars similar to our Sun for brightness changes produced by planetary transits.

The Kepler Mission came to an end after four years when two of the four reaction wheels, used to point the spacecraft, ceased to function. The Kepler Mission was reborn as the [K2 Mission](https://exoplanetarchive.ipac.caltech.edu/docs/K2Mission.html), which ran for an additional five years.

# TransitSense: GPT-Inspired Exoplanet Detection

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)


A novel approach to exoplanet detection that applies transformer architecture and next-value prediction (inspired by GPT) to stellar light curve analysis from NASA's Kepler Space Telescope.

## Overview

TransitSense treats stellar brightness measurements as sequential data, similar to how language models process text. By training a transformer exclusively on confirmed exoplanet host stars, the model learns to recognize "exoplanet-like" behavioral patterns. Stars that exhibit similar predictable characteristics become detection candidates.

**Key Innovation:** Instead of traditional transit detection alone, we combine next-value prediction accuracy with classical dip counting to create a composite detection score that reduces false positives.

## Performance

- **Overall Accuracy**: 97.0%
- **Exoplanet Detection Rate**: 40% (2/5 test samples)
- **False Positive Rate**: 2.3% (13/565 normal stars)
- **Precision**: 13% | **Recall**: 40% | **F1-Score**: 0.20

## Features

- **Transformer-based prediction**: 4-layer encoder with 8 attention heads (1.2M parameters)
- **Dual-signal detection**: Combines ML prediction error with classical transit signatures
- **Automatic threshold optimization**: F1-score based threshold selection
- **Interactive Streamlit dashboard**: Real-time detection and visualization
- **Comprehensive analysis**: Confusion matrices, score distributions, individual breakdowns
- **What Dataset we Have:**
- **Name**: Kepler Labelled Time Series Data, mostly the First 3 campaigns.
- **Files**: `exoTrain.csv` (250MB) and `exoTest.csv` (27.6MB)
- **Source**: NASA Kepler Space Telescope observations

## **What the Data Is:**

**Light curves** - measurements of star brightness over time (like a heartbeat monitor, but for stars)

- **Training**: 5,087 stars (37 with planets, 5,050 without)
- **Test**: 570 stars (5 with planets, 565 without)
- **Each star**: 3,197 brightness measurements taken over time
- **Labels**:
    - Label 1 = Normal star (no planet)
    - Label 2 = Star with confirmed exoplanet

## **How Exoplanet Detection Works:**

When a planet passes in front of its star (from our view), it blocks some light → **the star appears dimmer**. This creates a **periodic dip** in brightness.

`Normal star brightness:  ————————————————
Star with planet:        ————\__/————\__/————
                              ↑ transit dips`

![Transit_method.gif](attachment:916646e3-18c2-4ef5-9516-97731ddf09af:Transit_method.gif)

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/transitsense.git
cd transitsense

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

## Quick Start

### 1. Data Preparation

Download preprocessed Kepler data:

```bash
# Place kepler_processed.npz in data/ directory
# Contains X_train, X_test, y_train, y_test

```

### 2. Training

Train the transformer on exoplanet stars:

```bash
python train.py --epochs 20 --lr 0.001 --batch-size 4

```

### 3. Detection

Run detection on test set:

```bash
python detect.py --model models/exoplanet_signature_model.pt \
                 --data data/kepler_processed.npz \
                 --error-weight 0.35

```

### 4. Interactive Dashboard

Launch Streamlit app:

```bash
streamlit run app.py

```

## Project Structure

```
EXOPLANET-APP/
├── .streamlit/
├── .venv/
├── assets/
│   └── aetherion_logo_white.jpg
├── models/
│   ├── exoplanet_signature_model.pt
│   └── model.pkl
├── app.py
├── requirements.txt

NASASPACEAPPS/
│
├── venv/                                  # Virtual environment for dependencies
│
├── exoplanet_signature_model.pt           # PyTorch model (trained weights)
│
├── exoTrain.csv                           # Training dataset (CSV)
├── exoTest.csv                            # Testing dataset (CSV)
│
├── kepler-labelled-time-series-data.zip   # Raw Kepler data (compressed)
├── kepler_processed.npz                   # Preprocessed Kepler dataset (NumPy format)
│
├── trainmodel.py                                  

```

## Methodology

### Phase 1: Model Training

- Train transformer exclusively on 37 confirmed exoplanet host stars
- Objective: Next brightness value prediction (MSE loss)
- Learn temporal patterns characteristic of planet-hosting systems

### Phase 2: Feature Extraction

For each test star, compute:

1. **Prediction Error**: Mean squared error between predicted and actual brightness
2. **Transit Count**: Number of significant dips (>3σ below median)

### Phase 3: Composite Scoring

```python
score = 0.35 × error_score + 0.65 × transit_score

```

- Normalize both features to 0-100 scale
- Weight transit count more heavily (more reliable signal)

### Phase 4: Classification

- Optimize threshold using F1-score
- Flag stars above threshold as exoplanet candidates

## Model Architecture

```python
TimeSeriesTransformer(
    input_dim=1,
    d_model=128,
    nhead=8,
    num_layers=4,
    dim_feedforward=512,
    dropout=0.1,
    max_seq_len=3196
)

```

**Training Details:**

- Optimizer: AdamW (lr=0.001, weight_decay=0.01)
- Scheduler: ReduceLROnPlateau
- Gradient clipping: 1.0
- Training time: ~1h (NVIDIA RTX 4060 laptop)

## Results Analysis

### Detected Exoplanets (2/5)

- **Exoplanet #4**: Score 57.8 (250 dips, strong transit signature)
- **Exoplanet #5**: Score 62.4 (299 dips, strongest signal)

### Missed Exoplanets (3/5)

- **Exoplanet #1-3**: Scores 43.8-46.9 (96-130 dips, subtle transits)

### Key Insight

The model successfully detects exoplanets with frequent, deep transits but struggles with subtle cases, mirroring a fundamental astronomical challenge: the easiest discoveries have already been made.

## Dataset

**Source**: NASA Kepler Mission (Campaign 3 + augmented confirmed exoplanets)

**Training Set**: 5,087 stars

- 37 confirmed exoplanet hosts
- 5,050 normal stars

**Test Set**: 570 stars

- 5 confirmed exoplanet hosts
- 565 normal stars

**Light Curve Length**: 3,197 time-series measurements per star

## Requirements

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
streamlit>=1.28.0

```

## Future Work

- [ ]  Incorporate multi-scale temporal analysis
- [ ]  Ensemble with classical BLS algorithm
- [ ]  Expand training data with K2 and TESS missions
- [ ]  Add Bayesian uncertainty quantification
- [ ]  Implement active learning for iterative refinement

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## License

MIT License - see LICENSE file for details

## Citation

If you use this work, please cite:

```
@software{transitsense2025,
  title={TransitSense: GPT-Inspired Exoplanet Detection},
  author={Aetherion Lab},
  year={2025},
  url={https://github.com/manna3i/Aetherion-exoplanets}
}

```

## Acknowledgments

- NASA Kepler Mission for providing open astronomical data
- Kaggle community for curated dataset
- NASA Space Apps Hackathon 2025

## Contact

For questions or collaboration: adnanmanna3i@gmail.com
---

**Built for NASA Space Apps Hackathon 2025**
