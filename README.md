# 🧠 Le Super-Ensemble : Système de Recommandation Hybride Avancé

> **🏆 Performance Finale : MAP@10 = 0.21181**  
> *Classé 1er sur la Leaderboard Kaggle (17.5%)*

---

## 📖 1. La Vision du Projet

Ce projet ne se contente pas d'appliquer un algorithme standard. Il construit une **architecture "Super-Ensemble"** conçue pour capturer les nuances subtiles du comportement des lecteurs que les modèles traditionnels manquent.

### Le Défi "Cold Start" & "Long Tail"
Les approches classiques (Collaboratif pur) échouent sur les livres rares (Long Tail) ou les nouveaux utilisateurs. Les approches de contenu (S-BERT) manquent de précision sur les tendances virales.

### La Solution : Fusion de 5 Signaux Complémentaires

| Signal | Technique Utilisée | Librairie / Outil | Rôle |
|--------|-------------------|-------------------|------|
| **Sémantique** | Sentence-BERT | `sentence-transformers` (`all-MiniLM-L6-v2`) | Encode le texte (Titre + Auteur + Sujets) en vecteurs 384D |
| **Collaboratif** | TF-IDF + Time-Decay | `scikit-learn` (`TfidfTransformer`) | Pondère les interactions par récence ($e^{-\lambda t}$) |
| **Séquentiel** | Matrice de Co-visitation | `scipy.sparse` | Calcule $P(item_{next} \| item_{last})$ pour les séries |
| **Latent** | SVD (Factorisation) | `scipy.sparse.linalg.svds` | Réduit en 100 facteurs latents ($U \Sigma V^T$) |
| **Lexical** | BM25 / TF-IDF | `scikit-learn` (`TfidfVectorizer`) | Similarité exacte sur les mots-clés (titres, auteurs) |

**Autres techniques :**
- **Boost Re-buy** : Favorise les items déjà consultés (historique long-terme uniquement)
- **Boost Popularité** : Léger bonus pour les items "trending"
- **Cache Disque** : Embeddings S-BERT sauvegardés en `.npy` pour accélérer les relances

---

## ⚙️ 2. Architecture Technique

### Diagramme de Flux Simplifié

```
┌─────────────────────────────────────────────────────────────────┐
│                        DONNÉES D'ENTRÉE                         │
│  interactions_train.csv (u, i, t)  +  items.csv (métadonnées)   │
└─────────────────────────────────────────────────────────────────┘
                                 │
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
          ┌──────────┐    ┌──────────┐    ┌──────────┐
          │  S-BERT  │    │ TF-IDF   │    │   SVD    │
          │ Semantic │    │ Collab   │    │  Latent  │
          └────┬─────┘    └────┬─────┘    └────┬─────┘
               │               │               │
               └───────┬───────┴───────┬───────┘
                       ▼               ▼
              ┌─────────────┐   ┌─────────────┐
              │   BM25      │   │ Sequential  │
              │  Keywords   │   │ Next-Item   │
              └──────┬──────┘   └──────┬──────┘
                     │                 │
                     └────────┬────────┘
                              ▼
                 ┌────────────────────────┐
                 │   SCORE FINAL FUSIONNÉ │
                 │  + Boost Re-buy        │
                 │  + Boost Popularité    │
                 └───────────┬────────────┘
                             ▼
                    [ TOP-10 Recommandations ]
```

### Les Composants Mathématiques

#### A. Le Cœur Hybride (`alpha`)
Nous fusionnons une similarité collaborative ($S_{collab}$) et sémantique ($S_{sem}$) :
$$ S_{base} = \alpha \cdot S_{collab}(t) + (1-\alpha) \cdot S_{sem} $$
*   **Innovation** : $S_{collab}$ utilise une décroissance temporelle $e^{-\lambda \Delta t}$ avec deux demi-vies (1 jour et 250 jours) pour capturer à la fois l'humeur du moment et les goûts profonds.

#### B. Le "Game Changer" Séquentiel (`seq_weight`)
Bien que faible seul (MAP ~0.159), le modèle séquentiel capture une **orthogonalité** cruciale : la probabilité conditionnelle.
$$ P(i_{next} | i_{last}) \approx \log(1 + count(i_{last} \to i_{next})) $$
*Il corrige les erreurs du modèle sémantique qui peut recommander le Tome 3 avant le Tome 1.*

#### C. La Factorisation SVD (`svd_weight`)
Décomposition en valeurs singulières ($U \Sigma V^T$) de la matrice d'interactions.
*   **Rôle** : "Lissage" global. Il remplit les trous de la matrice sparse en connectant des communautés de lecteurs disjointes.

---

## 🧪 3. Journal des Expériences & Analyse Critique

Voici les résultats réels de nos itérations, montrant pourquoi l'Ensemble est nécessaire.

| # | Approche | MAP@10 | Statut | Analyse Critique |
|---|---|---|---|---|
| **★** | **Super-Ensemble (Production)** | **0.2118** | **WINNER** | **La somme est supérieure aux parties.** |
| 7 | Coupled Semantic (Naive) | 0.2045 | Échec | Dilution du signal "Re-buy" par le bruit court-terme. |
| 8 | Semantic Hybrid (ChatGPT) | 0.2040 | Mitigé | L'enrichissement aide, mais l'architecture est le facteur limitant. |
| 0 | BM25 Probabilistic | 0.1954 | Baseline | Excellent sur les titres exacts, aveugle au sens. |
| 4 | Diversification (Auteur) | 0.1940 | Échec | Forcer la diversité nuit à la précision pure (Trade-off). |
| 6 | Filtre Items Rares | 0.1631 | Échec | La "Long Tail" contient de la valeur prédictive cachée. |
| **9** | **Sequential / Co-visitation** | **0.1593** | **Pivot** | **Faible score seul, mais apporte +2% dans l'ensemble.** |
| 1 | EASE (Linear Model) | 0.1067 | Échec | Overfitting massif sur ce dataset sparse. |
| 3 | Ensemble Short/Long Term | 0.1038 | Échec | Sans sémantique, le collaboratif pur plafonne. |
| 2 | SVD (Latent Factors) | 0.0400 | Faible | Trop abstrait seul, mais excellent régularisateur. |

---

## 📂 4. Architecture Complète du Projet

### 4.1 Tableau des Fichiers et Dossiers

| Chemin | Type | Description |
|--------|------|-------------|
| **`data/`** | 📁 Dossier | Contient toutes les données d'entrée et de cache. |
| `data/interactions_train.csv` | 📄 CSV | Historique des interactions utilisateur-item (colonnes: `u`, `i`, `t`). |
| `data/items.csv` | 📄 CSV | Métadonnées des livres (Title, Author, Subjects, Publisher). |
| `data/items_enriched_ai_turbo.csv` | 📄 CSV | Items enrichis via GPT-4o-mini (description, clean_author, category). |
| `data/sample_submission.csv` | 📄 CSV | Format attendu pour la soumission Kaggle. |
| `data/eda/` | 📁 Dossier | Exports de l'analyse exploratoire (graphiques, stats). |
| `data/cache/` | 📁 Dossier | Cache disque des embeddings S-BERT (accélère les relances). |
| **`src/`** | 📁 Dossier | Code source Python principal. |
| `src/models/production.py` | 🐍 Python | **CLASSE MAÎTRESSE** : `SemanticHybridRecommender` (Super-Ensemble). |
| `src/models/experimental.py` | 🐍 Python | Laboratoire d'expériences : BM25, EASE, SVD, Sequential, etc. |
| `src/models/svd.py` | 🐍 Python | Wrapper pour la factorisation SVD (`scipy.sparse.linalg.svds`). |
| `src/models/base.py` | 🐍 Python | Classe abstraite `BaseRecommender` (interface commune). |
| `src/models/__init__.py` | 🐍 Python | Exports des classes de modèles. |
| `src/preprocessing.py` | 🐍 Python | `DataLoader` : chargement, nettoyage, mapping IDs, split temporel. |
| `src/metrics.py` | 🐍 Python | Calcul vectorisé du MAP@K (Mean Average Precision). |
| `src/tuning.py` | 🐍 Python | Script de Grid Search pour optimiser les hyperparamètres. |
| `src/EnrichissementChatGPT.py` | 🐍 Python | Enrichissement des métadonnées via API OpenAI (multithreadé). |
| **`notebooks/`** | 📁 Dossier | Notebooks Jupyter pour l'analyse et l'entraînement. |
| `notebooks/01_Data_Analysis.ipynb` | 📓 Notebook | EDA : distributions, sparsité, long-tail, visualisations. |
| `notebooks/02_Main_Model_Training.ipynb` | 📓 Notebook | **Pipeline principal** : entraînement + validation (MAP@10). |
| `notebooks/03_Main_Model_Submission_File_Generator.ipynb` | 📓 Notebook | Génération du fichier CSV de soumission Kaggle. |
| `notebooks/04_All_Experiments.ipynb` | 📓 Notebook | **Journal des expériences** : teste tous les modèles isolément. |
| **`app/`** | 📁 Dossier | Application de démonstration. |
| `app/streamlit_app.py` | 🐍 Python | Interface web Streamlit pour les bibliothécaires. |
| **`submission/`** | 📁 Dossier | Fichiers de soumission générés. |
| `submission/submission_final.csv` | 📄 CSV | Dernière soumission Kaggle (Top-10 par utilisateur). |
| `requirements.txt` | 📄 Texte | Liste des dépendances Python. |
| `README.md` | 📄 Markdown | Ce document. |

### 4.2 Comment Tester les Expériences

Pour reproduire et comparer toutes nos expériences (BM25, EASE, SVD, Sequential, etc.) :

```bash
# 1. Ouvrir le notebook d'expériences
jupyter notebook notebooks/04_All_Experiments.ipynb

# 2. Exécuter toutes les cellules
# Le notebook va :
#   - Charger les données et créer un split 80/20
#   - Entraîner chaque modèle expérimental isolément
#   - Afficher un tableau comparatif des scores MAP@10
```

**Structure du notebook `04_All_Experiments.ipynb` :**
1.  **Configuration** : Import des classes depuis `src/models/experimental.py`
2.  **Boucle d'expériences** : Chaque modèle est instancié, entraîné, et évalué
3.  **Synthèse** : Tableau final trié par performance

**Ajouter une nouvelle expérience :**
1.  Créer une nouvelle classe dans `src/models/experimental.py` (hériter de `BaseRecommender`)
2.  L'importer dans `src/models/__init__.py`
3.  Ajouter un appel `run_experiment(MaClasse, "Nom", **params)` dans le notebook

---

## 📈 5. Chronologie du Projet (2 Semaines)

| Jour | Phase | Activités | Score MAP@10 | Décision Clé |
|------|-------|-----------|--------------|--------------|
| **J1** | 🔍 Exploration | EDA, compréhension des données, statistiques de base | — | Identifier la sparsité (99.7%) et le problème Long-Tail |
| **J2** | 🔍 Exploration | Analyse des distributions temporelles, patterns de re-buy | — | Décider d'utiliser le Time-Decay |
| **J3** | 🛠️ Baseline | Implémentation TF-IDF collaboratif simple | 0.142 | Baseline fonctionnelle mais faible |
| **J4** | 🛠️ Itération | Ajout de S-BERT pour la similarité sémantique | 0.168 | +18% : le contenu aide significativement |
| **J5** | 🛠️ Itération | Fusion Hybride (α = 0.5) Collab + Sémantique | 0.182 | Synergie confirmée |
| **J6** | 🧪 Expériences | Test BM25, EASE, SVD isolés | 0.04-0.19 | Aucun modèle seul ne dépasse l'hybride |
| **J7** | 🛠️ Itération | Ensemble multi-demi-vies (1j + 250j) | 0.195 | Capturer court-terme ET long-terme |
| **J8** | 🧪 Expériences | Test diversification, filtrage items rares | 0.16-0.19 | Contraintes = perte de précision |
| **J9** | 💡 Breakthrough | Découverte du boost "Re-buy Décorrélé" | 0.201 | Ne pas diluer les favoris historiques |
| **J10** | 🧪 Expériences | Enrichissement ChatGPT des métadonnées | 0.204 | Amélioration marginale |
| **J11** | 💡 Breakthrough | **Implémentation Sequential (Co-visitation)** | **0.211** | **+5% : Le Game Changer !** |
| **J12** | 🔧 Tuning | Grid Search sur tous les poids (α, seq, bm25, svd) | 0.2118 | Paramètres optimaux trouvés |
| **J13** | 📦 Finalisation | Nettoyage du code, documentation, tests | 0.2118 | Code prêt pour production |
| **J14** | 🚀 Livraison | Soumission Kaggle + Rédaction README | **0.2118** | **🏆 1ère Place !** |

---

## 🚀 6. Guide d'Utilisation

### Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Reproduire la Performance (0.211)
1.  Ouvrez `notebooks/02_Main_Model_Training.ipynb`.
2.  Exécutez toutes les cellules.
3.  Le modèle s'entraînera avec les hyperparamètres optimaux (`alpha=0.5`, `half_life=[1, 250]`, `seq_weight=0.3`).

### Générer la Soumission Kaggle
```bash
jupyter notebook notebooks/03_Main_Model_Submission_File_Generator.ipynb
```
*Génère `submission/submission_final.csv` entraîné sur 100% des données.*

---

## 🖥️ 7. Application Streamlit — BiblioRec

Une application web complète a été développée pour permettre aux **bibliothécaires** d'utiliser le modèle de recommandation de manière intuitive.

### 7.1 Lancement

```bash
# Depuis la racine du projet
streamlit run app/streamlit_app.py
```

L'application s'ouvre automatiquement dans votre navigateur à l'adresse `http://localhost:8501`.

### 7.2 Fonctionnalités par Onglet

| Onglet | Icône | Fonctionnalité | Cas d'Usage |
|--------|-------|----------------|-------------|
| **Accueil** | 🏠 | Dashboard avec KPIs + Recherche rapide | Vue d'ensemble de la bibliothèque |
| **Recommander à un Usager** | 👤 | Recommandations personnalisées pour un usager existant | "Que proposer à l'usager #1234 ?" |
| **Découverte (Cold Start)** | 🔍 | Recherche sémantique par description textuelle | Nouvel usager : "J'aime les thrillers nordiques" |
| **Livres Similaires** | 📖 | Trouver des livres similaires à un titre donné | "Quoi lire après Harry Potter ?" |
| **Statistiques** | 📊 | Graphiques : Top auteurs, distribution, sparsité | Analyse de la collection |
| **À Propos** | ℹ️ | Documentation technique du modèle | Comprendre l'algorithme |

### 7.3 Guide d'Utilisation Détaillé

#### 👤 Recommander à un Usager (Cas Principal)

1. **Sélectionner l'usager** : Recherchez par ID ou parcourez la liste
2. **Configurer les options** :
   - ☑️ *Exclure les livres déjà empruntés* (recommandé)
   - 📚 *Max par auteur* : Limite la redondance (ex: max 2 livres du même auteur)
   - 💡 *Afficher les explications* : Montre pourquoi chaque livre est recommandé
3. **Cliquer sur "Générer les recommandations"**
4. **Résultats** :
   - Liste des Top-K livres avec titre, auteur, explication
   - Bouton 👍 pour donner du feedback (sauvegardé dans `data/feedback_log.csv`)
   - Bouton 📥 pour télécharger la liste en CSV

#### 🔍 Découverte — Cold Start

Pour les **nouveaux usagers** sans historique :
1. Décrivez leurs goûts en texte libre :
   > *"Romans policiers scandinaves, ambiance sombre, enquêtes psychologiques"*
2. Le modèle S-BERT encode cette description et trouve les livres les plus proches sémantiquement

#### 📊 Statistiques

- **Top 15 Auteurs** : Graphique horizontal des auteurs les plus empruntés
- **Distribution des emprunts** : Histogramme du nombre d'emprunts par usager
- **Jauges** :
  - *Sparsité* : % de la matrice User×Item qui est vide (~99.7%)
  - *Couverture* : % des livres ayant au moins 1 emprunt
  - *Usagers actifs* : % des usagers ayant au moins 1 emprunt

### 7.4 Configuration (Sidebar)

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| **Alpha** | 0.5 | Balance Collaboratif ↔ Sémantique (0 = 100% sémantique) |
| **Demi-vie court** | 1 jour | Capture les tendances immédiates |
| **Demi-vie long** | 250 jours | Capture les goûts de fond |
| **Top-K** | 10 | Nombre de recommandations à afficher |
| **Métadonnées enrichies** | Off | Utiliser `items_enriched_ai_turbo.csv` (GPT-4) |

### 7.5 Captures d'Écran (Description)

```
┌─────────────────────────────────────────────────────────────┐
│  📚 BiblioRec — Système de Recommandation Intelligent       │
│  Propulsé par le Super-Ensemble                             │
├─────────────────────────────────────────────────────────────┤
│  [🏠 Accueil] [👤 Usager] [🔍 Découverte] [📖 Similaires]   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│   │  12,847  │  │  15,123  │  │  98,456  │  │   7.6    │   │
│   │ Usagers  │  │  Livres  │  │ Emprunts │  │ Moy/User │   │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│                                                             │
│   🔎 Recherche Rapide: [_________________]                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📚 8. Références Bibliographiques

Notre approche s'appuie sur des travaux de recherche reconnus dans le domaine des systèmes de recommandation :

| Concept | Référence |
|---------|-----------|
| **Sentence-BERT** | Reimers & Gurevych (2019) - *"Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"* - EMNLP |
| **Time-Decay Collaborative** | Koren (2009) - *"Collaborative Filtering with Temporal Dynamics"* - KDD |
| **BM25** | Robertson & Zaragoza (2009) - *"The Probabilistic Relevance Framework: BM25 and Beyond"* - Foundations and Trends in IR |
| **SVD pour RecSys** | Funk (2006) - *"Netflix Update: Try This at Home"* - Blog post (Netflix Prize) |
| **Ensemble Methods** | Bell & Koren (2007) - *"Lessons from the Netflix Prize Challenge"* - SIGKDD Explorations |
| **Hybrid Recommenders** | Burke (2002) - *"Hybrid Recommender Systems: Survey and Experiments"* - User Modeling and User-Adapted Interaction |

---

## 🔬 9. Limites & Perspectives

### Limites Actuelles

| Limite | Description | Impact |
|--------|-------------|--------|
| **Cold Start Utilisateurs** | Nouveaux utilisateurs sans historique | Fallback sur popularité uniquement |
| **Cold Start Items** | Nouveaux livres sans interactions | Dépend uniquement de S-BERT (contenu) |
| **Biais de Popularité** | Items populaires sur-représentés | Peut nuire à la découverte (sérendipité) |
| **Scalabilité Mémoire** | Matrice de similarité S-BERT dense (N×N) | Limite pratique ~100K items |
| **Données Implicites** | Pas de feedback négatif explicite | On ne sait pas ce que l'utilisateur n'aime PAS |

### Améliorations Futures

1. **Graph Neural Networks** : Intégrer LightGCN ou PinSage pour mieux capturer les relations utilisateur-item dans un graphe
2. **Transformers Séquentiels** : Remplacer la co-visitation par SASRec ou BERT4Rec pour une modélisation séquentielle plus fine
3. **Multi-Objectif** : Optimiser simultanément précision + diversité + nouveauté
4. **A/B Testing** : Valider les gains offline (MAP@10) par des métriques online (CTR, temps de lecture)
5. **Explicabilité** : Ajouter des justifications ("Recommandé car vous avez aimé X")

---

## 📊 10. Choix de la Métrique : Pourquoi MAP@10 ?

### Comparaison des Métriques de Ranking

| Métrique | Formule Simplifiée | Avantage | Inconvénient |
|----------|-------------------|----------|--------------|
| **MAP@K** | Moyenne des précisions aux positions de hit | Pénalise les erreurs en haut du ranking | Ignore la diversité |
| **NDCG@K** | Gain pondéré par $\log_2(position)$ | Pondération plus fine | Plus complexe à interpréter |
| **Recall@K** | $\frac{\|hits\|}{\|relevant\|}$ | Simple et intuitif | Ignore totalement l'ordre |
| **MRR** | $\frac{1}{rang_{premier\_hit}}$ | Focus sur le 1er résultat | Ignore les autres positions |
| **Hit Rate@K** | 1 si au moins 1 hit, 0 sinon | Très simple | Trop binaire |

### Notre Choix : MAP@10

$$MAP@K = \frac{1}{|U|} \sum_{u \in U} \frac{1}{\min(K, |R_u|)} \sum_{k=1}^{K} P(k) \cdot rel(k)$$

- **Standard Kaggle/RecSys** : Permet la comparaison avec d'autres équipes
- **Équilibre Précision/Ordre** : Récompense les bons items ET leur position
- **K=10** : Correspond à une page de résultats typique (UX réaliste)

---

## 👥 11. Auteurs & Crédits

**Université Paris 1 Panthéon-Sorbonne — Master TIDE**

| Membre | Contributions |
|--------|---------------|
| **Sacha Jocic** | Architecture modèle, Tuning hyperparamètres, Pipeline SVD/Séquentiel |
| **Léa Jouffrey** | Analyse de données (EDA), Logique métier, Application Streamlit |
| **Saloua Dekhissi** | Enrichissement sémantique (ChatGPT), Tests expérimentaux, Documentation |

---

*Ce projet est l'aboutissement de 2 semaines de recherche intensive sur les systèmes de recommandation hybrides, inspiré par les solutions gagnantes du Netflix Prize et des compétitions RecSys.*
