import subprocess, sys, os

# ── Resolve project root & set working directory ──
# This makes all relative paths (data/, models/, etc.) work
# regardless of where the script is launched from (local or Kaggle).
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
from models.beat_cnn import build_beat_cnn
from training.callbacks import make_callbacks
# from utils.balancing import get_class_weights
from utils.oversampling import moderate_ros

X = np.load("data/processed/beat_X.npy") 
y = np.load("data/processed/beat_y.npy") 

# Shape of original processed data
print(X.shape, y.shape)
print(np.unique(y, return_counts=True))

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2)

# Apply moderate ROS only on training set to balance the dataset
X_train_resampled, y_train_resampled = moderate_ros(X_train, y_train)

print("After ROS:", np.unique(y_train_resampled, return_counts=True))

# Downsample overrepresented classes
class_indices = {cls: np.where(y_train_resampled == cls)[0] for cls in np.unique(y_train_resampled)}

drop_config = {
    0: 20000,   # majority class (Normal beats 'N')
    1: 40000,   # This is usually the hardest class in all datasets because it has relatively fewer samples and more similarity to other classes
    2: 240000,  # Class 'V' (Ventricular ectopic beats)
    3: 90000,   # Fusion beats are tricky because they overlap with normal/ventricular morphologies (Fusion beats 'F')
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

model = build_beat_cnn()
model.fit(
    X_train_resampled, y_train_resampled,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=96,
    # class_weight = get_class_weights(y_train), we don't use it since we are using ROS
    callbacks=make_callbacks("beat")
)

os.makedirs("models", exist_ok=True)
model.save("models/beat_final.h5")













