"""
Verification du boost dynamique pour SemanticHybridRecommender.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import sparse

# Rendre robustes les imports
try:
    from src.models import SemanticHybridRecommender
    from src.preprocessing import DataLoader
    from src.metrics import mapk_score
except ImportError:
    sys.path.append('.')
    from models import SemanticHybridRecommender
    from preprocessing import DataLoader
    from metrics import mapk_score

# CONFIG
THIS_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'data'))

# BEST PARAMS FOUND (Baseline)
ALPHA = 0.5
WEIGHTS = [0.5, 0.5] # Back to optimal
HALF_LIVES = [1, 250]

# Testing BM25
BM25_WEIGHT = 0.2

print("🚀 Lancement Verification BM25 Boost...")
loader = DataLoader(f'{DATA_DIR}/interactions_train.csv', f'{DATA_DIR}/items.csv')
train_df, val_df = loader.get_time_split(train_ratio=0.8)

# Validation Matrix
val_rows = val_df['u_idx'].values
val_cols = val_df['i_idx'].values
val_data = np.ones(len(val_df))
val_matrix = sparse.csr_matrix(
    (val_data, (val_rows, val_cols)),
    shape=(loader.n_users, loader.n_items)
)

# Model
model = SemanticHybridRecommender(loader.n_users, loader.n_items)
model.fit(
    train_df, 
    loader.items_df, 
    alpha=ALPHA, 
    half_life_days=HALF_LIVES, 
    ensemble_weights=WEIGHTS
)

print(f"\nTesting Prediction with BM25={BM25_WEIGHT} (ReBuy={0.5}, Pop={0.2})...")
preds = model.predict(k=10, batch_size=2000, re_buy_factor=0.5, pop_factor=0.2, bm25_weight=BM25_WEIGHT)
score = mapk_score(preds, val_matrix, k=10)

print(f"🎯 FINAL MAP@10 SCORE: {score:.5f}")
