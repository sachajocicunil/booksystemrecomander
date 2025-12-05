# src/metrics.py
import numpy as np

def mapk_score(predicted_indices, true_matrix, k=10):
    """
    Calcule le Mean Average Precision @K (MAP@K).

    Paramètres
    ----------
    predicted_indices : array-like de shape (n_users, K)
        Pour chaque utilisateur u (index interne), la liste des K indices d'items `i_idx` prédits, ordonnés par score décroissant.
    true_matrix : array-like ou scipy.sparse
        Matrice binaire Vérité-Terrain de shape (n_users, n_items). Une valeur >0 en (u, i) indique que l'item i est
        réellement pertinent pour l'utilisateur u dans le set de validation.
    k : int
        Coupure K (par défaut 10). Doit être cohérent avec la taille de `predicted_indices`.

    Notes
    -----
    - Les utilisateurs sans vérités terrain (aucun item positif en validation) sont ignorés dans la moyenne.
    - La précision moyenne pour un utilisateur est la moyenne des précisions aux rangs où des hits surviennent,
      normalisée par min(nombre d'items pertinents, K).
    - `predicted_indices` doit être indexé en `i_idx` (indices internes), pas en IDs réels.

    Retour
    ------
    float
        MAP@K moyen sur les utilisateurs considérés.
    """
    scores = []
    num_users = true_matrix.shape[0]

    for u in range(num_users):
        if hasattr(true_matrix, "indices"):
            actual = true_matrix[u].indices
        else:
            actual = np.where(true_matrix[u] > 0)[0]

        if len(actual) == 0:
            continue

        pred = predicted_indices[u]
        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(pred):
            if p in actual:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        scores.append(score / min(len(actual), k))

    return np.mean(scores)