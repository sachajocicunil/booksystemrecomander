"""
Grid search simple pour SemanticHybridRecommender.
Peut être exécuté depuis le dossier racine ou depuis src/ (les imports s'adaptent).
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import sparse

# Rendre robustes les imports selon le cwd
try:
    from src.models import SemanticHybridRecommender
    from src.preprocessing import DataLoader
    from src.metrics import mapk_score
except ImportError:
    # Fallback si on exécute depuis src/
    sys.path.append('.')
    from models import SemanticHybridRecommender
    from preprocessing import DataLoader
    from metrics import mapk_score

# --- CONFIGURATION ---
# DATA_DIR résolu relativement à ce fichier pour éviter les surprises de cwd
THIS_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'data'))

# 1. Paramètres affectant le FIT (Coûteux)
alphas = [0.4, 0.5, 0.6, 0.7]

# [Poids Court Terme, Poids Long Terme]
weights_options = [
    [0.5, 0.5],
    [0.6, 0.4],
    [0.8, 0.2],
]

HALF_LIVES = [1, 250]  # Fixé pour l'instant

# 2. Paramètres affectant le PREDICT (Rapide)
re_buy_factors = [0.5, 1.0, 1.5, 2.0]
pop_factors = [0.0, 0.05, 0.1, 0.2]

print("🚀 Lancement du Grid Search Avancé...")
loader = DataLoader(f'{DATA_DIR}/interactions_train.csv', f'{DATA_DIR}/items.csv')
train_df, val_df = loader.get_time_split(train_ratio=0.8)

# Matrice de validation
val_rows = val_df['u_idx'].values
val_cols = val_df['i_idx'].values
val_data = np.ones(len(val_df))
val_matrix = sparse.csr_matrix(
    (val_data, (val_rows, val_cols)),
    shape=(loader.n_users, loader.n_items)
)

# On instancie le modèle
model = SemanticHybridRecommender(loader.n_users, loader.n_items)

# Pré-chargement S-BERT (une seule fois pour chauffer le cache)
print("Chargement initial S-BERT...")
model.fit(train_df, loader.items_df, alpha=0.5, half_life_days=HALF_LIVES)

best_score = -1
best_params = {}

total_combinations = len(alphas) * len(weights_options) * len(re_buy_factors) * len(pop_factors)
current_iter = 0

print(f"\n--- Début des tests ({total_combinations} combinaisons) ---")

# Boucle EXTERNE : Paramètres qui nécessitent un re-training (fit)
for alpha in alphas:
    for w in weights_options:
        
        # On refait le fit
        print(f"\n[FIT] Alpha={alpha}, Weights={w}")
        model.fit(
            train_df,
            loader.items_df,
            alpha=alpha,
            half_life_days=HALF_LIVES,
            ensemble_weights=w
        )
        
        # Boucle INTERNE : Paramètres de prédiction uniquement
        for rb in re_buy_factors:
            for pop in pop_factors:
                current_iter += 1
                
                # Prediction avec paramètres dynamiques
                preds = model.predict(k=10, batch_size=2000, re_buy_factor=rb, pop_factor=pop)
                score = mapk_score(preds, val_matrix, k=10)
                
                print(f"  ({current_iter}/{total_combinations}) ReBuy={rb}, Pop={pop} -> MAP@10: {score:.5f}")
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        'alpha': alpha,
                        'weights': w,
                        're_buy_factor': rb,
                        'pop_factor': pop
                    }
                    print(f"  🔥 Nouveau Record! {score:.5f}")

print("\n" + "=" * 30)
print(f"🏆 MEILLEURE CONFIGURATION TROUVÉE")
print(f"Score : {best_score:.5f}")
print(f"Alpha : {best_params['alpha']}")
print(f"Weights: {best_params['weights']}")
print(f"Re-Buy Factor : {best_params['re_buy_factor']}")
print(f"Pop Factor : {best_params['pop_factor']}")
print("=" * 30)
