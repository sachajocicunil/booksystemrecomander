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

# Grille de paramètres à tester
# On teste des alpha autour de 0.5
alphas = [0.2,0.3,0.4,0.5, 0.6, 0.7]

# On teste différents équilibres Court Terme (CT) / Long Terme (LT)
# [Poids CT, Poids LT]
weights_options = [
    [0.7, 0.3],  #
    [0.5, 0.5],  # Equilibré
    [0.6, 0.4],
    [0.8, 0.2],  # Très focus sur le récent
    [0.9, 0.1]
]

# Half-lives fixés (car recalculer les matrices prend trop de temps)
HALF_LIVES = [30, 150]

print("🚀 Lancement du Grid Search (Tuning)...")
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

# On instancie le modèle UNE SEULE FOIS pour charger S-BERT
model = SemanticHybridRecommender(loader.n_users, loader.n_items)

# On pré-calcule les matrices internes pour ne pas tout refaire à chaque boucle
# C'est une astuce pour aller vite : on appelle fit() une fois pour tout charger
print("Pré-calcul des matrices de base...")
model.fit(train_df, loader.items_df, alpha=0.5, half_life_days=HALF_LIVES)

# On récupère les composants internes pour simuler le predict avec différents poids
# (Hack pour éviter de refaire le 'fit' qui est long)
# Note: Cela suppose que la méthode fit() a stocké 'ensemble_models' et 'sim_content'
# Si ton modèle est strictement encapsulé, on va faire la boucle standard (plus lent mais sûr)

best_score = -1
best_params = {}

print(f"\n--- Début des tests ({len(alphas) * len(weights_options)} combinaisons) ---")

for alpha in alphas:
    for w in weights_options:
        print(f"Testing Alpha={alpha}, Weights={w}...", end=" ")

        # On refait le fit rapide (S-BERT est déjà en cache mémoire dans la classe si bien géré,
        # sinon ce sera un peu long mais supportable)
        model.fit(
            train_df,
            loader.items_df,
            alpha=alpha,
            half_life_days=HALF_LIVES,
            ensemble_weights=w
        )

        # Prédiction (Batch plus petit pour aller vite)
        preds = model.predict(k=10, batch_size=2000)
        score = mapk_score(preds, val_matrix, k=10)

        print(f"-> MAP@10: {score:.5f}")

        if score > best_score:
            best_score = score
            best_params = {'alpha': alpha, 'weights': w}

print("\n" + "=" * 30)
print(f"🏆 MEILLEURE CONFIGURATION TROUVÉE")
print(f"Score : {best_score:.5f}")
print(f"Alpha : {best_params['alpha']}")
print(f"Weights: {best_params['weights']}")
print("=" * 30)