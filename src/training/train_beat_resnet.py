import subprocess, sys, os

# ── Resolve project root & set working directory ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

# ── Auto-install imbalanced-learn if missing (needed on Kaggle) ──
try:
    import imblearn  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'imbalanced-learn', '-q'])

import numpy as np
from sklearn.model_selection import train_test_split
from models.beat_resnet import build_beat_resnet # Import the new model
from training.callbacks import make_callbacks
from utils.oversampling import moderate_ros

# 1. Load Data
X = np.load("data/processed/beat_X.npy") 
y = np.load("data/processed/beat_y.npy") 

# 2. Split
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 3. Apply your ROS + Downsampling Logic
X_train_resampled, y_train_resampled = moderate_ros(X_train, y_train)

# 4. Downsample overrepresented classes
class_indices = {cls: np.where(y_train_resampled == cls)[0] for cls in np.unique(y_train_resampled)}

drop_config = {
    0: 30000,   # majority class (Normal beats 'N')
    1: 20000,   # This is usually the hardest class in all datasets because it has relatively fewer samples and more similarity to other classes
    2: 240000,  # Class 'V' (Ventricular ectopic beats)
    3: 70000,   # Fusion beats are tricky because they overlap with normal/ventricular morphologies (Fusion beats 'F')
    4: 300000,  # Unknown/Other 'Q'
}

np.random.seed(42)
drop_indices = []

for cls, n_to_drop in drop_config.items():
    indices = class_indices[cls]
    if len(indices) >= n_to_drop:
        dropped = np.random.choice(indices, size=n_to_drop, replace=False)
        drop_indices.extend(dropped)
    else:
        print(f"Class {cls}: only {len(indices)} samples available, cannot drop {n_to_drop}")

drop_indices = np.array(drop_indices)
keep_indices = np.setdiff1d(np.arange(len(y_train_resampled)), drop_indices)

X_train_resampled = X_train_resampled[keep_indices]
y_train_resampled = y_train_resampled[keep_indices]

print("After downsampling:", np.unique(y_train_resampled, return_counts=True))

# 5. Build and Train
model = build_beat_resnet(input_shape=(360, 1))

model.fit(
    X_train_resampled, y_train_resampled,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=make_callbacks("beat_resnet")
)
os.makedirs("models/resnet", exist_ok=True)
model.save("models/beat_final.h5")
