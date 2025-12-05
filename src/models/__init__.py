# src/models/__init__.py
from .production import SemanticHybridRecommender
from .experimental import (
    BM25Recommender,
    EASERecommender,
    SVDRecommender,
    DiversifiedRecommender,

    EnsembleHybridRecommender,    # Nouveau
    HistoryFilterRecommender,     # Nouveau
    LowInteractionRecommender,
    CoupledSemanticRecommender,
    SemanticHybridRecommenderChatGPT,
    SequentialRecommender
)