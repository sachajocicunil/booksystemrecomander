# src/models/base.py
from abc import ABC, abstractmethod
import numpy as np

class BaseRecommender(ABC):
    """
    Classe abstraite pour uniformiser tous les modèles de recommandation.
    Garantit que chaque modèle possède une méthode fit() et predict().
    """
    def __init__(self, n_users, n_items):
        self.n_users = n_users
        self.n_items = n_items

    @abstractmethod
    def fit(self, df_interactions, df_items, **kwargs):
        """Entraînement du modèle"""
        pass

    @abstractmethod
    def predict(self, k=10, batch_size=1000):
        """
        Prédiction des top K items pour chaque user.
        Doit retourner une liste de listes ou un np.array (n_users, k).
        """
        pass