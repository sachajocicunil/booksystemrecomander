# src/preprocessing.py
import pandas as pd
import numpy as np

class DataLoader:
    """
    Charge les données et prépare les mappings entre IDs réels et indices internes.

    Paramètres
    -----------
    path_interactions : str
        Chemin vers `interactions_train.csv` (colonnes attendues: `u`, `i`, `t`).
    path_items : str
        Chemin vers `items.csv` ou `items_enriched_ai_turbo.csv` (doit contenir `i` et idéalement
        `Title`, `Author`, `Subjects`).

    Attributs
    ---------
    interactions : pd.DataFrame
        Interactions dédupliquées avec colonnes ajoutées `u_idx`, `i_idx`.
    items_df : pd.DataFrame
        Items alignés sur les `i` présents dans les interactions, avec `i_idx`.
    u_map / i_map : dict
        Mapping ID réel -> index interne.
    idx_to_i : dict
        Mapping inverse index interne -> ID réel (utile pour soumissions).
    n_users / n_items : int
        Tailles des espaces utilisateurs et items.
    """
    def __init__(self, path_interactions, path_items):
        self.interactions = pd.read_csv(path_interactions)
        # Nettoyage de base
        self.interactions = self.interactions.drop_duplicates()
        self.interactions['u'] = self.interactions['u'].astype(int)
        self.interactions['i'] = self.interactions['i'].astype(int)

        # Mappings (ID réel -> Index matrice)
        self.u_unique = sorted(self.interactions['u'].unique())
        self.i_unique = sorted(self.interactions['i'].unique())

        self.u_map = {uid: idx for idx, uid in enumerate(self.u_unique)}
        self.i_map = {iid: idx for idx, iid in enumerate(self.i_unique)}
        self.idx_to_i = {idx: iid for iid, idx in self.i_map.items()}

        # Ajout des index au dataframe
        self.interactions['u_idx'] = self.interactions['u'].map(self.u_map)
        self.interactions['i_idx'] = self.interactions['i'].map(self.i_map)

        # Ajout i_idx aux items pour faciliter les tris plus tard
        self.items_df = pd.read_csv(path_items)
        # On suppose que items.csv a une colonne 'i' correspondant à l'ID item
        if 'i' in self.items_df.columns:
            self.items_df['i_idx'] = self.items_df['i'].map(self.i_map)
            # On ne garde que les items qui existent dans les interactions
            self.items_df = self.items_df.dropna(subset=['i_idx'])
            self.items_df['i_idx'] = self.items_df['i_idx'].astype(int)

        self.n_users = len(self.u_unique)
        self.n_items = len(self.i_unique)

    def get_time_split(self, train_ratio=0.8):
        """Divise en Train/Val basé sur le temps (rank)"""
        df = self.interactions.sort_values(['u_idx', 't']).copy()
        df['rank'] = df.groupby('u_idx')['t'].rank(pct=True, method='dense')

        train = df[df['rank'] < train_ratio].copy()
        val = df[df['rank'] >= train_ratio].copy()
        return train, val

    def get_full_data(self):
        """Retourne tout le dataset pour l'entrainement final"""
        return self.interactions.copy()