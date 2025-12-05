# Système de Recommandation — Documentation du Projet

Ce dépôt implémente un système de recommandation hybride "collaboratif + sémantique" pour des items (livres), avec un pipeline complet : EDA, entraînement, évaluation (MAP@K), tuning de paramètres, enrichissement sémantique des métadonnées via LLM, et génération de fichier de soumission.

Ce README explique l’architecture, où trouver chaque composant, comment exécuter les notebooks/scripts, et comment reproduire les résultats.

---

## 1) Arborescence du projet

```
PythonProject2/
├── data/                       # Données d’entrée et auxiliaires
│   ├── interactions_train.csv  # Interactions utilisateur–item (colonnes attendues: u, i, t)
│   ├── items.csv               # Métadonnées des items (Title, Author, Subjects, ...)
│   ├── items_enriched_ai_turbo.csv  # (optionnel) Items enrichis via LLM
│   └── sample_submission.csv # (optionnel) Format d’IDs utilisateurs cible pour la soumission
│   └── eda/
│       ├── interactions_by_day.csv                 #output eda table
│       ├── item_popularity_counts.csv              #output eda table
│       └── user_interaction_counts.csv             #output eda table
├── notebooks/
│   ├── 01_Data_Analysis.ipynb                  # EDA complète des données
│   ├── 02_Main_Model_Training.ipynb            # Entraînement + évaluation du modèle de prod
│   ├── 03_Main_Model_Submission_File_Generator.ipynb # Génération du CSV de soumission
│   └── 04_All_Experiments.ipynb                # Expériences non concluantes ou abandonnées
├── src/
│   ├── preprocessing.py         # Chargement des données, mapping d’IDs, splits temporels
│   ├── metrics.py               # Métriques d’évaluation (MAP@K)
│   ├── tuning.py                # Grid search des hyperparamètres clés
│   ├── EnrichissementChatGPT.py # Script d’enrichissement sémantique (OpenAI)
│   └── models/
│       ├── base.py              # Classe abstraite BaseRecommender
│       ├── production.py        # Modèle de production: SemanticHybridRecommender
│       ├── experimental.py      # Modèles alternatifs/ablation/essais
│       └── __init__.py          # Exporte les classes de modèles
├── submission/
│   └── (sorties)                # Fichiers de soumission générés (ex: submission_final.csv)
├── requirements.txt             # Dépendances Python du projet
└── README.md                    # Ce document
```

---

## 2) Données attendues

- interactions_train.csv (obligatoire)
  - Colonnes minimales: `u` (user id), `i` (item id), `t` (timestamp/ordre)
  - Types: `u` et `i` entiers; `t` numérique (ex: epoch sec) ou ordre croissant par utilisateur
- items.csv (recommandé)
  - Colonnes utiles: `i` (id item), `Title`, `Author`, `Subjects`
  - Ces champs servent à la représentation sémantique (S-BERT) pour le contenu
- items_enriched_ai_turbo.csv (optionnel)
  - Généré par `src/EnrichissementChatGPT.py`, ajoute `description`, `clean_author`, `category`
  - Peut améliorer la composante sémantique
- sample_submission.csv (optionnel)
  - Colonne `user_id` pour dicter l’ordre des lignes du fichier final de prédictions

Remarque: `src/preprocessing.py` filtre et aligne `items` avec les `i` présents dans `interactions` et ajoute `i_idx` (index interne) pour le matching.

---

## 3) Composants principaux

### 3.1 Prétraitement — `src/preprocessing.py`
- Classe `DataLoader`:
  - Lit les CSV, déduplique les interactions, cast `u` et `i` en int.
  - Crée des mappings: `u_map`, `i_map`, `idx_to_i`, et ajoute `u_idx`/`i_idx` aux interactions.
  - Aligne `items` sur les `i` connus et ajoute `i_idx`.
  - Expose `n_users`, `n_items`, `items_df`.
  - Splits temporel par utilisateur via `get_time_split(train_ratio=0.8)` (rang percentile par `t`).
  - `get_full_data()` pour récupérer tout le dataset (entraînement final).

### 3.2 Modèle de production — `src/models/production.py`
- Classe `SemanticHybridRecommender(BaseRecommender)`
  - Idée: Hybride "collaboratif + sémantique" avec décroissance temporelle (time-decay), TF‑IDF user profiles, similarité cosinus, et fusion avec similarité sémantique issue de S‑BERT.
  - Découplage du signal de re-buy: le boost de ré-achat est appliqué uniquement sur l’historique long-terme pour ne pas diluer les items anciens.
  - Étapes clés de `fit`:
    1) Encode le texte item via `SentenceTransformer('all-MiniLM-L6-v2')` en combinant `Title`, `Author`, `Subjects`.
    2) Pour chaque demi-vie (`half_life_days`) demandée:
       - Calcule poids temporels par interaction; construit matrice sparse utilisateurs×items.
       - TF‑IDF pour profils utilisateurs.
       - Similarité collaborative items×items via cosinus sur profils transposés.
       - Fusion par `alpha`: `alpha * sim_collab + (1 - alpha) * sim_content`.
    3) Conserve la matrice utilisateur du plus grand half-life comme historique long-terme.
    4) Calcule une popularité globale (somme long-terme) pour un léger boost.
  - `predict(k, batch_size)`
    - Agrège les scores des sous-modèles (poids `ensemble_weights`).
    - Ajoute un boost re-buy basé sur l’historique long-terme (facteur ~1.5).
    - Ajoute un léger boost de popularité (0.1 * pop normalisée).
    - Sélectionne le Top‑K par utilisateur.

Paramètres efficaces d’après expérimentations:
- `alpha ≈ 0.5`
- `half_life_days = [1, 250]` (très court vs très long)
- Des poids d’ensemble non-uniformes peuvent sur-apprendre localement (cf. commentaire sur overfitting dans les notebooks)

### 3.3 Modèles expérimentaux — `src/models/experimental.py`
- Implémente des variantes/utilitaires pour tests et ablations:
  - `BM25Recommender`: pondération BM25 sur interactions, similarité cosinus.
  - `EASERecommender`: méthode EASE (régularisée) items×items.
  - `SVDRecommender`: factorisation basique (SVD tronquée) sur matrice utilisateurs×items.
  - `DiversifiedRecommender`: diversification simple des recommandations.
  - `EnsembleHybridRecommender`: variante d’ensemble.
  - `HistoryFilterRecommender`: filtrage sur historique.
  - `LowInteractionRecommender`: stratégie pour faibles interactions.
  - `CoupledSemanticRecommender` et `SemanticHybridRecommenderChatGPT`: variantes sémantiques couplées.

Toutes héritent de `BaseRecommender` (voir `src/models/base.py`, interface `fit`/`predict`).

### 3.4 Métriques — `src/metrics.py`
- `mapk_score(predicted_indices, true_matrix, k=10)`
  - Calcule MAP@K pour l’évaluation hors-ligne avec une matrice vérité-terrain (CSR ou dense).

### 3.5 Tuning — `src/tuning.py`
- Grid search simple sur:
  - `alpha ∈ [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]`
  - `ensemble_weights ∈ {[0.7,0.3],[0.5,0.5],[0.6,0.4],[0.8,0.2],[0.9,0.1]}` avec `HALF_LIVES=[30,150]`
- Utilise `DataLoader` pour split temporel; évalue `MAP@10` sur validation.

### 3.6 Enrichissement LLM — `src/EnrichissementChatGPT.py`
- Enrichit `items.csv` en batch via l’API OpenAI (modèle `gpt-4o-mini`) pour générer:
  - `description` (FR), `clean_author`, `category`.
- Sauvegarde `items_enriched_ai_turbo.csv`.
- Paramètres batch et threads contrôlent la vitesse; penser aux limites de rate.
- Important: définir la variable d’environnement `OPENAI_API_KEY` et vérifier la facturation. Les champs enrichis ne sont utilisés que si vous remplacez `items.csv` par le fichier enrichi lors de l’entraînement/soumission.

---

## 4) Notebooks — Guide rapide

1. `01_Data_Analysis.ipynb`
   - EDA complète: qualité des données, distributions, récence, densité, long-tail, etc.
2. `02_Main_Model_Training.ipynb`
   - Pipeline de base: chargement via `DataLoader`, split temporel 80/20, entraînement `SemanticHybridRecommender`, prédiction `Top‑10`, calcul `MAP@10`.
   - Exemple de configuration performante: `alpha=0.5`, `half_life_days=[1, 250]`.
3. `03_Main_Model_Submission_File_Generator.ipynb`
   - Entraînement sur toutes les données puis génération du CSV de soumission (format Kaggle: `user_id,recommendation`).
   - Gère les cold‑start avec fallback sur les items populaires.
   - Peut utiliser `items_enriched_ai_turbo.csv` si disponible.
4. `04_All_Experiments.ipynb`
   - Journal d’expériences infructueuses ou abandonnées pour la traçabilité.

---

## 5) Installation & Pré‑requis

- Python 3.9+ recommandé
- GPU non requis (S‑BERT mini fonctionne sur CPU) mais accélère l’encodage si disponible

Dépendances principales (à ajouter dans `requirements.txt` si besoin):
- numpy, scipy, pandas, scikit-learn
- sentence-transformers
- tqdm
- openai (si enrichissement LLM)
- matplotlib, seaborn (pour EDA)

Exemple d’installation rapide:
```
python -m venv .venv
source .venv/bin/activate  # Ou .venv\Scripts\activate sous Windows
pip install --upgrade pip
pip install numpy scipy pandas scikit-learn sentence-transformers tqdm openai matplotlib seaborn
```

---

## 6) Exécution — Étapes clés

### 6.1 Évaluation locale (notebook d’entraînement)
- Ouvrir `notebooks/02_Main_Model_Training.ipynb`
- Vérifier les chemins `../data/*.csv`
- Lancer toutes les cellules pour entraîner et afficher `MAP@10`

### 6.2 Tuning
Depuis `src/` (ou ajuster les imports selon IDE):
```
python src/tuning.py
```
- Le script charge les données, fait un split, lance le grid search et affiche la meilleure config.

### 6.3 Entraînement complet + Soumission
- Ouvrir `notebooks/03_Main_Model_Submission_File_Generator.ipynb`
- Configurer:
  - `DATA_DIR = '../data'`
  - `BEST_ALPHA = 0.5`
  - `BEST_HALFLIFE = [1, 250]` (ou vos meilleurs hyperparamètres)
- Exécuter le notebook pour produire `submission/submission_final.csv`

### 6.4 Enrichissement des items (optionnel)
```
export OPENAI_API_KEY=your_api_key_here
python src/EnrichissementChatGPT.py
# Fichier de sortie: data/items_enriched_ai_turbo.csv
```
- Ensuite, dans vos notebooks, remplacez `items.csv` par `items_enriched_ai_turbo.csv` lors du chargement via `DataLoader`.

---

## 7) Détails du format d’évaluation

- Construction de la vérité-terrain validation: on crée une matrice CSR de taille `(n_users, n_items)` avec `1` aux positions `(u_idx, i_idx)` présentes dans `val_df`.
- `mapk_score` itère par utilisateur et calcule la moyenne de la précision cumulée jusqu’à `K`.
- Important: `predict` retourne des indices d’items internes (`i_idx`). Pour générer des IDs réels, convertir via `loader.idx_to_i`.

---

## 8) Bonnes pratiques & pièges courants

- S‑BERT: nécessite `sentence-transformers`. Le premier encodage peut télécharger le modèle.
- Mémoire: l’encodage de tous les items peut être coûteux si `items` est très grand. Utilisez un modèle léger (`all‑MiniLM‑L6‑v2`, par défaut) et éventuellement batcher/mettre en cache.
- Overfitting local: des poids d’ensemble (CT/LT) trop orientés vers le set de validation peuvent baisser le score public.
- Cold‑start: le générateur de soumission gère les utilisateurs inconnus via un fallback populaire.
- Alignement IDs: toujours passer des `items` alignés (avec `i_idx`) au modèle; `DataLoader` s’en charge si les colonnes attendues sont présentes.
- OpenAI API: stocker la clé de manière sécurisée (variables d’environnement), respecter les politiques d’utilisation. Ne pas committer de clés.

---

## 9) Réplication rapide (cheat‑sheet)

1) Préparer `data/interactions_train.csv` et `data/items.csv`.
2) Lancer `02_Main_Model_Training.ipynb` pour vérifier que `MAP@10` est raisonnable.
3) Optionnel: `python src/tuning.py` pour affiner `alpha` et les poids.
4) Entraîner sur tout + générer soumission via `03_Main_Model_Submission_File_Generator.ipynb`.
5) Optionnel: enrichir `items` via `src/EnrichissementChatGPT.py` et réentraîner.

---

## 10) Licence & Crédits

- Code des modèles inspiré de méthodes classiques (TF‑IDF, EASE, SVD) et de l’encodeur `sentence-transformers`.
- Auteur: Sacha Jocic,Léa Jouffrey, Saloua Dekhissi

Ce README a été créé pour documenter l’implémentation et faciliter la prise en main.


---

## 11) Résultats (Académiques & Compétition)

### 11.1 Score de compétition
- Plateforme: Kaggle (défi local)
- Score Public: `MAP@10 = 0.17127`
- Rang: `1er`

### 11.2 Résultats locaux (validation temporelle 80/20)
Réévaluations sur le même split temporel; toutes les approches ci‑dessous sont inférieures au modèle final, sauf mention contraire.

| # | Approche                             | MAP@10   | Statut                         |
|---|--------------------------------------|----------|--------------------------------|
| 7 | Coupled Semantic (Re-buy Dilution)   | 0.204524 | Inférieur au Best Model        |
| 8 | Semantic Hybrid (ChatGPT Enriched)   | 0.203973 | Inférieur au Best Model        |
| 3 | Ensemble Short/Long Term             | 0.202633 | Inférieur au Best Model        |
| 0 | BM25 Probabilistic                   | 0.195390 | Inférieur au Best Model        |
| 4 | Diversification (Max 2/Author)       | 0.194026 | Inférieur au Best Model        |
| 6 | Filtre Items Rares (<3 vues)         | 0.163061 | Inférieur au Best Model        |
| 1 | EASE (Linear Model)                  | 0.106677 | Inférieur au Best Model        |
| 2 | SVD (Latent Factors)                 | 0.040002 | Inférieur au Best Model        |
| 5 | Filtre Strict Historique             | 0.020747 | Inférieur au Best Model        |

- Modèle final (Production — SemanticHybridRecommender, décorrélé re‑buy) : `MAP@10 local = 0.20691`

### 11.3 Progression dans le temps (timeline synthétique)
Note: cette timeline est reconstituée a posteriori pour le rapport; les jalons et scores intermédiaires sont indicatifs.

| Semaine | Jalons principaux                                   | Config clé                          | MAP@10 val |
|---------|-----------------------------------------------------|-------------------------------------|------------|
| D1      | EDA, baseline TF‑IDF users × cosinus                | decay=fixe (HL=30)                  | 0.142      |
| D2      | Ajout S‑BERT (contenu) + fusion α                   | α=0.5, HL=30                        | 0.182      |
| D3      | Ensemble de demi‑vies (CT/LT)                       | HL=[30,150], poids=[0.5,0.5]        | 0.201      |
| D4      | Analyse re‑buy; correction «decoupled re‑buy boost» | HL=[1,250], boost re‑buy long‑terme | 0.206      |
| D5      | Tentatives diversification/contraintes              | —                                   | 0.19–0.20  |
| D6      | Enrichissement LLM (items)                          | Metadata LLM                        | 0.204      |
| D6      | Nettoyage + soumission finale                       | HL=[1,250], α=0.5                   | 0.20691    |
| D7      | Streamlit app                                       | ------                              | pretty     |

— Ces résultats montrent que la décorrélation du boost de re‑achat par rapport à l’ensemble CT/LT est la modification déterminante.

---

## 12) Reproductibilité & Bonnes pratiques (MAJ)

- Fixer les seeds n’a pas d’impact majeur ici (pas de composantes stochastiques dans le pipeline par défaut), mais reste recommandé pour l’EDA/échantillonnages.
- Les encodages S‑BERT peuvent être mis en cache selon votre environnement pour accélérer les itérations.
- Pour l’enrichissement LLM:
  - Définir la clé API via variable d’environnement: `export OPENAI_API_KEY=...`
  - Le script lit désormais la clé depuis `OPENAI_API_KEY` et sauvegarde dans `data/items_enriched_ai_turbo.csv`.
  - Respecter les limites de taux (MAX_WORKERS et BATCH_SIZE ajustables) et les politiques d’usage.

## 13) Limites & Travaux futurs

- Long‑tail: les items très rares restent difficiles; une régularisation spécifique ou des embeddings supervisés pourraient aider.
- Popularité: le léger boost global est fixe; on pourrait l’apprendre ou le conditionner au segment utilisateur.
- Sérendipité/diversité: l’ajout de contraintes naïves nuit au MAP@K; envisager des objectifs multi‑critères.
- Cold‑start utilisateurs: stratégies basées sur signaux contextuels (heure, device) ou profils proxy non exploitées ici.



## 14) Version finale — Spécifications et paramètres (production.py + 02_Main_Model_Training.ipynb)

Cette section précise exactement ce que réalise la version finale livrée du modèle de production et récapitule les paramètres utilisés dans le notebook d’entraînement principal.

### 14.1 Modèle final: `SemanticHybridRecommender` (src/models/production.py)
- Type: Hybride « collaboratif + sémantique » avec décroissance temporelle, ensemble multi–demi‑vies, et boost de re‑achat décorrélé (appliqué seulement sur l’historique long‑terme).
- Entrées attendues de `fit(df_interactions, df_items, ...)`:
  - `df_interactions` avec colonnes: `u_idx`, `i_idx`, `t` (timestamp/ordre). Des duplicats sont possibles, pondérés ensuite par la récence.
  - `df_items` aligné avec les `i_idx` et colonnes textuelles: `Title`, `Author`, `Subjects` (vides acceptées, remplies par défaut par des chaînes vides avant encodage).

- Étapes internes de `fit` (spécification):
  1) Encodage sémantique items via `SentenceTransformer('all-MiniLM-L6-v2')` sur la concaténation « `Title`. `Author`. `Subjects` »; calcul d’une similarité de contenu `sim_content` par cosinus des embeddings.
  2) Pour chaque demi‑vie `hl ∈ half_life_days`:
     - Calcul des poids temporels: `weight = exp(- ln(2) / hl * days_diff)`, avec `days_diff = (last_user_ts - t) / (24*3600)` par utilisateur.
     - Construction d’une matrice sparse utilisateurs×items pondérée par `weight`.
     - Transformation TF‑IDF (L2, IDF lissé) des profils utilisateurs.
     - Similarité collaborative items×items par cosinus sur la transposée des profils (`user_profile.T`).
     - Fusion par `alpha`: `sim_final = alpha * sim_collaborative + (1 - alpha) * sim_content`.
     - Stockage du sous‑modèle dans l’ensemble avec son poids `ensemble_weights[j]` (si non fourni: uniforme).
     - Conservation de la matrice utilisateur du plus grand `hl` comme « historique long‑terme » (`long_term_user_matrix`).
  3) Calcul d’une popularité globale: somme des colonnes de `long_term_user_matrix`, normalisée sur [0,1].

- Étapes internes de `predict(k, batch_size)`:
  1) Agrégation des scores des sous‑modèles: pour chaque batch d’utilisateurs, on calcule `scores_j = user_batch · sim_final_j`, puis `final_scores = Σ_j weight_j * scores_j`.
  2) Re‑buy boost (décorrélé): `final_scores += 1.5 * long_term_user_matrix_batch` (applique le boost uniquement depuis l’historique long‑terme pour ne pas diluer les vieux favoris).
  3) Légère popularité: `final_scores += 0.1 * pop_scores` (même vecteur pour tous les utilisateurs du batch).
  4) Sélection Top‑K: `argpartition` pour extraire les K meilleurs indices `i_idx`, puis tri local par score décroissant.

- Remarques importantes:
  - Le boost « 1.5 » et le terme de popularité « 0.1 » sont fixes dans la version livrée (et documentés ici pour la reproductibilité).
  - La fusion `alpha` intervient uniquement dans la construction des matrices de similarité items×items (exploration); le re‑buy boost est ajouté après, à l’étape de scoring (exploitation).
  - Les demi‑vies très éloignées (p. ex. `[1, 250]`) permettent de couvrir à la fois l’ultra‑récent et l’historique très long.

### 14.2 Paramètres utilisés dans `02_Main_Model_Training.ipynb`
- Split temporel: `train_ratio = 0.8` (par utilisateur).
- Modèle: `SemanticHybridRecommender(n_users=loader.n_users, n_items=loader.n_items)`.
- Appel d’entraînement:
  ```text
  # Exemple d'appel (illustratif) :
  # 1) Charger les données et splitter (train/val)
  # 2) Instancier SemanticHybridRecommender(n_users, n_items)
  # 3) Appeler fit(alpha=0.5, half_life_days=[1, 250]) puis predict(k=10)
  # 4) Évaluer via MAP@10
  ```
- Prédiction: `preds = model.predict(k=10, batch_size=1000)`.
- Évaluation: `MAP@10` via `mapk_score(preds, val_matrix, k=10)`.

- Valeurs et conventions à retenir (version livrée):
  - `alpha = 0.5`
  - `half_life_days = [1, 250]`
  - `ensemble_weights = None` (donc uniformes)
  - `k = 10`
  - `batch_size = 1000`
  - `re_buy_boost = 1.5` (ajouté après l’agrégation des sous‑modèles)
  - `popularity_boost = 0.1` (coefficient appliqué au vecteur de popularité normalisé)

- Notes pratiques:
  - L’option `ensemble_weights=[0.6, 0.4]` a montré un meilleur score local mais a dégradé le score Kaggle public, indiquant un sur‑apprentissage au split de validation — elle est donc laissée en commentaire dans le notebook.
  - Pour la génération de soumission (`03_Main_Model_Submission_File_Generator.ipynb`), la même configuration est utilisée sur l’ensemble des données (entraînement intégral), avec fallback « items populaires » pour les utilisateurs cold‑start.


---

## 15) Application Streamlit — Démonstrateur pour Bibliothécaire

Cette application autonome permet d’explorer le catalogue et de générer des recommandations avec le modèle final (`SemanticHybridRecommender`). Elle vise un usage « pratico‑pratique » par un/une bibliothécaire.
Dispo en ligne sur: https://booksystemrecomandersachaleasaloua.streamlit.app/

### 15.1 Installation & Lancement
- Prérequis: dépendances déjà listées dans `requirements.txt` (inclut `streamlit`).
- Installation (si besoin):
  ```
  pip install -r requirements.txt
  ```
- Lancer l’application depuis la racine du dépôt:
  ```
  streamlit run app/streamlit_app.py
  ```
- Par défaut, l’app cherchera les fichiers:
  - `data/interactions_train.csv`
  - `data/items_enriched_ai_turbo.csv` (si présent) ou `data/items.csv`

Astuce: les chemins des CSV sont modifiables dans la barre latérale.

### 15.2 Fonctionnalités principales
- Overview
  - Statistiques rapides (utilisateurs, items, interactions)
  - Recherche rapide par titre/auteur/sujet
- Recommend for Patron (utilisateur connu)
  - Sélection d’un `user_id` et génération du Top‑K
  - Règles « métier » optionnelles: exclure déjà‑lus, limiter à N livres par auteur
  - Export CSV des recommandations et recueil de feedback (like) journalisé dans `data/feedback_log.csv`
- Cold‑Start (Text Search)
  - Recommandations à partir d’une description libre (mots‑clés, auteur, genre)
  - Utilise l’encodeur sémantique S‑BERT
- Similar Books (item‑item)
  - Trouver des livres similaires à un livre choisi (voisinage sémantique/collaboratif)
- Analytics
  - Auteurs les plus consultés (bar chart), popularité, histogrammes de récence
- Settings
  - Visualisation des chemins de données, génération d’un CSV de soumission d’exemple (aperçu)

### 15.3 Paramètres du modèle (barre latérale)
- `alpha` (collaboratif vs contenu), `half_life_days` (séparés par des virgules), `Top‑K`
- Bouton « Rebuild model (fit) » pour refitter avec les nouveaux paramètres
- Bouton « Clear caches » pour réinitialiser les caches Streamlit

Valeurs par défaut: `alpha=0.5`, `half_life_days=[1, 250]`, `k=10`, cohérentes avec la version finale documentée.

### 15.4 Performance & Caching
- Le premier lancement peut être long (téléchargement du modèle S‑BERT et encodage des items)
- Des caches (`st.cache_data`/`st.cache_resource`) évitent les recomputations coûteuses
- Un GPU n’est pas requis mais accélère S‑BERT si disponible

### 15.5 Dépannage
- « sentence‑transformers introuvable »: installez via `pip install -r requirements.txt`
- Colonne manquante dans `items.csv`: l’app tolère l’absence de `Author`/`Subjects` (remplacement par chaîne vide), mais la qualité sémantique est meilleure avec ces colonnes
- Mémoire: si le catalogue est très grand, envisagez de réduire temporairement `Top‑K` et de vérifier que `items.csv` ne contient pas de colonnes excessives

### 15.6 Sécurité & Données
- L’app lit uniquement des CSV locaux; aucun envoi de données n’est réalisé
- Les feedbacks sont stockés en local dans `data/feedback_log.csv`


### 15.7 Streamlit too heavy → Persist item embeddings to disk

I implemented robust on‑disk caching for S‑BERT item embeddings so the app no longer recomputes them every run. This dramatically reduces CPU/RAM usage and startup time.



#### What changed
- `src/models/production.py`
  - Added disk cache for item embeddings during `fit(...)`:
    - Builds a stable text "soup" per item (`Title`. `Author`. `Subjects`) sorted by `i_idx`.
    - Hashes content (`md5`) together with model name (`all-MiniLM-L6-v2`).
    - Loads embeddings from `data/cache/embeddings_{model}_{hash}.npy` if present; otherwise encodes once and saves `.npy` + `.json` metadata.
  - More robust handling when `Title`/`Author`/`Subjects` columns are missing (falls back to empty strings).

- `app/streamlit_app.py`
  - Caching upgraded for app‑side semantic features (Cold‑Start search, Similar Books):
    - `compute_item_embeddings(items_df, use_disk_cache=True)` uses the same disk cache scheme.
    - New sidebar toggle: “Use disk cache for embeddings” (default ON).
    - New sidebar action: “Clear embedding disk cache” to delete cached `.npy`/`.json` files.
    - Existing “Clear in‑memory caches” still clears Streamlit caches.

#### Why this fixes the issue
- The embeddings are the heavy step; now they’re encoded once and reused across sessions/runs.
- On subsequent app starts, the app directly memory‑maps or loads the saved `.npy` instead of recomputing, saving minutes and RAM spikes.

#### How to use
1) Install deps: `pip install -r requirements.txt`.
2) Run the app: `streamlit run app/streamlit_app.py`.
3) Ensure the sidebar toggle “Use disk cache for embeddings” is ON (default).
4) First run computes and saves to `data/cache/`; next runs should be instant.
5) If you change `items` content significantly, the cache automatically invalidates via content hash.
6) To force a rebuild, click “Clear embedding disk cache” (and optionally “Rebuild model (fit)”).

#### Notes
- Cache paths: `data/cache/embeddings_all-MiniLM-L6-v2_<md5>.npy` + `.json`.
- The production model (`fit`) and the Streamlit app both use compatible caching, so training and UI share the same saved embeddings if the items content matches.
- If the dataset is extremely large, consider keeping the disk cache on a fast drive (SSD) for best load times.

## 15.8 Amélioration de secours au cas ou qqn nous bats dans kaggle, on en a encore dans le sac
# 📈 Analyse et Stratégies d'Amélioration pour le Système de Recommandation (Compétition Kaggle)

Cette analyse propose une série d'améliorations visant à maximiser la performance du modèle de recommandation en production (combinant time-decay, TF-IDF collaboratif, S-BERT sémantique, et boosts spécifiques) en vue d'une augmentation du score **MAP@10** (Mean Average Precision at 10).

L'objectif est d'identifier des leviers d'optimisation, classés par retour sur investissement (ROI) estimé, pour consolider ou améliorer le classement dans la compétition.

---

## I. Optimisations Fines des Hyperparamètres (Quick Wins)
*Ces ajustements, nécessitant un faible effort, se concentrent sur la calibration des paramètres d'agrégation et d'exploration.*

| Amélioration | Description de l'Action | Justification Théorique | Δ MAP@10 Estimé |
| :--- | :--- | :--- | :--- |
| **1. Affinement de l'exposant de décélération ($α$)** | Élargissement de la grille de recherche de l'exposant $α$ (e.g., `{0.35, 0.45, 0.55, 0.65}`) et introduction d'une **demi-vie temporelle médiane** (e.g., 30 jours) au mélange existant (1 et 250 jours). | Les schémas de consommation intermédiaires peuvent capturer une mémoire sémantique utile. Un $α$ optimisé maximise la pertinence des interactions récentes. | `+0.002` à `+0.006` |
| **2. Calibration des coefficients de boost** | Balayage systématique des coefficients de **réachat** (*re-buy*) et de **popularité** (*pop*) autour des valeurs nominales. <br> Ex: $re\_buy \in \{1.2, 1.35, 1.65\}$ ; $pop \in \{0.05, 0.15\}$. | Assurer un équilibre optimal entre l'exploitation des préférences existantes et la découverte, évitant le surapprentissage sur l'historique utilisateur. | `+0.001` à `+0.003` |
| **3. Introduction d'un écart temporel (no-leakage gap)** | Suppression d'une petite fenêtre d'interactions (e.g., 2−3%) autour de la frontière temporelle entre l'ensemble d'entraînement et de validation. | Réduit le risque de fuite d'information (*temporal leakage*) pour les interactions immédiatement consécutives. Rend le score local plus fidèle au Test Public. | **Score local plus robuste** |

---

## II. Amélioration de la Qualité et Stabilité des Similarités Inter-Items
*La stabilité et la robustesse de la matrice de similarité inter-items sont critiques, en particulier face au bruit et aux long-tails.*

| Amélioration | Description de l'Action | Justification Théorique | Δ MAP@10 Estimé |
| :--- | :--- | :--- | :--- |
| **4. Élagage (Pruning) Top-K et Réduction de Bruit (Shrinkage)** | 1. **Top-K:** Conserver uniquement les $K$ voisins (e.g., $K \in \{100, 200, 400\}$).<br>2. **Shrinkage:** Appliquer la formule :<br> $$sim_{shrink} = sim \times \frac{n_{cooc}}{n_{cooc} + \lambda}$$ <br> où $\lambda$ est un hyperparamètre de régularisation. | Le Top-K minimise l'impact des similarités bruitées. Le shrinkage pondère à la baisse les similarités issues d'un faible support statistique ($n_{cooc}$ faible), stabilisant les items rares. | `+0.002` à `+0.010` |
| **5. Pondération de fusion ($α$) adaptative** | Utiliser un $\alpha_{eff}$ tel que $\alpha_{eff} = r \times \alpha$, où $r$ dépend de la densité de co-occurrence ($n_{cooc}$) ou par segmentation en bacs. | Dans les cas de faible preuve collaborative (*cold/rare items*), le modèle sémantique (S-BERT) prend le relais. Permet une fusion dynamique adaptée au contexte. | `+0.002` à `+0.006` |

---

## III. Amélioration de l'Encodage Sémantique (Contenu)
*L'optimisation des représentations vectorielles par S-BERT est un levier de gain "sûr".*

| Amélioration | Description de l'Action | Justification Théorique | Δ MAP@10 Estimé |
| :--- | :--- | :--- | :--- |
| **6. Pondération des champs de contenu (Text Soup)** | Assigner des poids aux champs pour l'encodeur. <br>Ex: $2 \times Titre + 1 \times Auteur + 1 \times Sujets + 0.5 \times Desc$. | Le Titre est souvent plus informatif et concis (signal fort), tandis qu'une description longue peut introduire du bruit, justifiant un poids moindre. | `+0.002` à `+0.006` |
| **7. Exploration d'encodeurs sémantiques alternatifs** | Tester des modèles S-BERT alternatifs pré-entraînés pour la recherche sémantique (ex: `multi-qa-MiniLM-L6-cos-v1`, `all-MiniLM-L12-v2`). | Ces modèles sont optimisés sur des tâches de similarité inter-document très proches de l'objectif final, capturant mieux les nuances sémantiques. | `+0.002` à `+0.008` |

---

## IV. Adaptation Personnalisée par Segments d'Utilisateurs
*L'hétérogénéité de la population nécessite une personnalisation au-delà du simple produit scalaire.*

| Amélioration | Description de l'Action | Justification Théorique | Δ MAP@10 Estimé |
| :--- | :--- | :--- | :--- |
| **8. $α$ adaptatif par profil d'activité** | Segmenter les utilisateurs (Léger $\le 3$, Moyen $4-20$, Dense $>20$). Assigner un $\alpha$ spécifique (ex: $\{0.2, 0.4, 0.6\}$). | Les utilisateurs "denses" bénéficient des signaux collaboratifs ($\alpha$ élevé), tandis que les "légers" (*cold-start*) reposent plus sur le contenu sémantique ($\alpha$ faible). | `+0.003` à `+0.012` |

---

## V. Re-classement (Re-ranking) Modéré et Robustesse
*Ajustements post-score pour améliorer l'expérience utilisateur sans dégrader la métrique.*

| Amélioration | Description de l'Action | Justification Théorique | Δ MAP@10 Estimé |
| :--- | :--- | :--- | :--- |
| **9. Pénalité douce de l'Auteur dans le Top-K** | Appliquer une pénalité soustractive modérée ($\gamma \in \{0.01, 0.02\}$) si un auteur est sur-représenté (ex: $\ge 3$ fois) dans le Top-K. | Offre une diversité contrôlée (*Exposure Fairness*) sans imposer une contrainte binaire rigide qui dégrade la précision. | **Neutre à Légèrement positif** |

---

## VI. Stratégie de Validation et Stabilité
*Essentiel pour éviter le sur-ajustement local.*

| Amélioration | Description de l'Action | Justification Théorique |
| :--- | :--- | :--- |
| **10. Validation Croisée Temporelle Glissante (Rolling CV)** | Exécuter sur $\ge 3$ tranches temporelles ($Train T_n \to Val T_n \to T_{n+\Delta}$). Valider si gain stable ($\ge 2/3$ folds) et significatif ($\ge 0.003$). | Simule l'évolution temporelle du jeu de données Kaggle, garantissant que les hyperparamètres ne sont pas spécifiques à une unique période ("lucky shot"). |
| **11. Politique Cold-Start par popularité récente** | Utiliser la popularité sur une fenêtre récente (30−60 jours) au lieu de globale pour les nouveaux utilisateurs. | Reflète mieux les tendances actuelles (trends) que la popularité historique globale. |

---

## VII. Plan de Priorisation (Ordre de Test Recommandé)

1.  **Ajustements Rapides :** $α$ et demi-vie médiane (I-1) → Calibration des boosts (I-2) → Élagage Top-K (II-4).
2.  **Qualité et Contenu :** Pondération du *text soup* (III-6) → Encodeur alternatif (III-7) → $α$ adaptatif par segment (IV-8) → Shrinkage/Fusion adaptative (II-4/II-5).
3.  **Robustesse :** Application du *Rolling CV* (VI-10) pour valider les gains → Test du re-ranking doux (V-9).

### 🎯 Objectif Réaliste de Gain
L'addition de 3 à 5 de ces leviers, s'ils sont bien calibrés et démontrent un gain stable en Rolling CV, devrait permettre un gain de **+0.006 à +0.020** en MAP@10 local.
