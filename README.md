# Détection de sources dans les questions parlementaires

Ekaterina LEONOVA — Master 1 Linguistique Informatique, Université Grenoble Alpes, 2025-2026

## 1. Introduction

Ce projet met en place un pipeline de classification binaire de phrases issues des questions écrites à l'Assemblée Nationale (XVIe législature) : une phrase contient-elle une référence à une source (rapport, étude, statistique, organisme…) ou non ?

Il s'agit d'une tâche de classification supervisée à deux classes, réalisée avec CamemBERT-base fine-tuné via Hugging Face Transformers. Le projet explore également l'apport de l'**Active Learning** pour réduire le coût de l'annotation : est-il possible d'améliorer les performances du modèle en ajoutant intelligemment des exemples au set d'entraînement, plutôt qu'aléatoirement ?

Le pipeline couvre : classification binaire de base, expérimentations Active Learning avec trois stratégies (random, uncertainty, KMeans), et évaluation des résultats.

---

## 2. Structure du projet

```
.
├── data/
│   ├── train.json                       
│   ├── valid.json                        
│   ├── test.json                         
│   └── predictions_with_context.json     # Pool pour Active Learning (~10 000 exemples)
├── results/
│   ├── results_random.json               # Résultats stratégie random (6 itérations)
│   └── results_uncertainty.json          # Résultats stratégie uncertainty (6 itérations)
│   └── results_kmeans.json          # Résultats stratégie kmeans (6 itérations)
├── classification_binaire.py             # Classification binaire
├── active_learning_final.py              # Active Learning — 3 stratégies
├── requirements.txt
└── README.md
```

---

## 3. Dépendances

| Bibliothèque | Usage |
|---|---|
| Python 3.9+ | Langage principal |
| torch | Backend PyTorch pour l'entraînement |
| transformers | CamemBERT, tokenizer, Trainer, TrainingArguments |
| datasets | Dataset HuggingFace |
| scikit-learn | Métriques, KMeans |
| sentencepiece | Tokenizer SentencePiece (requis par CamemBERT) |
| accelerate | Optimisation entraînement HuggingFace |
| numpy | Calculs numériques |
| matplotlib / seaborn | Visualisation (matrice de confusion) |

Installation :

```bash
pip install -r requirements.txt
```

---

## 4. Exécution

### Données requises

Placer les fichiers de données dans le dossier `data/`, ou adapter `BASE_DIR` / `TRAIN_PATH` dans chaque script.

### Classification binaire

```bash
python classification_binaire.py
```

Ou ouvrir `classification_binaire.ipynb` dans Colab (activer le GPU T4 avant de lancer).

### Active Learning

```bash
# Adapter BASE_DIR dans le script avant de lancer
python active_learning_final.py --strategy random
python active_learning_final.py --strategy uncertainty
python active_learning_final.py --strategy kmeans
```

Les résultats sont sauvegardés après chaque itération dans `results_<strategy>.json`.

Les expériences Active Learning ont été conduites sur le serveur de l'université (`miai-server.u-ga.fr`, GPU NVIDIA RTX A6000).

---

## 5. Description du corpus

Les données sont issues du projet [DAPEPANE](https://gricad-gitlab.univ-grenoble-alpes.fr/m1-projet-s2/benjamin-nikita-lihui-sonia/dapepane) — questions écrites à l'Assemblée Nationale (XVIe législature), annotées au format IOB pour la détection de références à des sources.

### Format des fichiers JSONL

Chaque ligne est un objet JSON représentant une phrase :

```json
{
  "uid": "QANR5L16QE16390",
  "tokens": ["Selon", "le", "rapport", "du", "GIEC", "..."],
  "labels": ["O", "O", "O", "O", "B-SRC", "..."],
  "prev": "phrase précédente dans le document",
  "next": "phrase suivante dans le document"
}
```

Les étiquettes IOB utilisées : `O` (hors source), `B-SRC` (début de source), `I-SRC` (suite de source).

Pour la classification binaire, les étiquettes IOB sont converties en label binaire : `1` si au moins un token porte `B-SRC` ou `I-SRC`, `0` sinon.

### Dataset 

Annotation réalisée manuellement dans le cadre du projet DAPEPANE.

| Split | Total | Classe 0 | Classe 1 |
|-------|-------|----------|----------|
| Train | 1500 | 542 (36%) | 958 (64%) |
| Validation | 200 | 72 (36%) | 128 (64%) |
| Test | 90 | 33 (37%) | 57 (63%) |


### Pool (Active Learning)

Le pool contient des phrases non annotées manuellement, avec des labels **automatiques** (bruités) produits par un modèle entraîné sur le dataset gold. Il est utilisé pour simuler un scénario d'Active Learning où l'on dispose d'un large stock d'exemples non labellisés.

| Fichier | Exemples | Classe 0 | Classe 1 |
|---------|----------|----------|----------|
| predictions_with_context.json | 10 000 | 8692 (87%) | 1308 (13%) |

Le fort déséquilibre du pool (87 % / 13 %) reflète la distribution naturelle du corpus parlementaire : la plupart des phrases ne contiennent pas de référence à une source.

---

## 6. Étape 1 — Classification binaire

### Modèle

CamemBERT-base fine-tuné pour la classification de séquences (`CamembertForSequenceClassification`, 2 classes). La loss utilisée est une Cross-Entropy pondérée (classe 0 = 1.0, classe 1 = 1.5) pour compenser le léger déséquilibre des classes.

### Hyperparamètres

| Paramètre | Valeur |
|-----------|--------|
| Modèle de base | camembert-base |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Max length | 256 tokens |
| Époques | 3 |
| Seuil de décision | 0.75 |

Le seuil de décision à 0.75 (au lieu de 0.5) favorise la précision sur la classe 1 en réduisant les faux positifs.

### Résultats sur le test (dataset gold)

| Métrique | Valeur |
|----------|--------|
| Accuracy | 0.967 |
| F1 (classe 1) | 0.964 |
| Précision (classe 1) | 0.982 |
| Rappel (classe 1) | 0.947 |

### Matrice de confusion

|  | Prédit 0 | Prédit 1 |
|--|----------|----------|
| **Réel 0** | 32 | 1 |
| **Réel 1** | 2 | 55 |

Le modèle commet très peu d'erreurs : 1 faux positif et 2 faux négatifs sur 90 exemples. Ces résultats très élevés dès le premier entraînement soulevaient la question de l'utilité de l'Active Learning sur ce corpus.

---

## 7. Étape 2 — Active Learning

### Principe

L'Active Learning vise à réduire le coût de l'annotation en sélectionnant intelligemment les exemples les plus informatifs à annoter parmi un pool non labellisé. Au lieu d'annoter aléatoirement, on choisit les exemples sur lesquels le modèle est le plus incertain — ce sont ceux qui apporteront le plus d'information lors du réentraînement.

À chaque itération, **k=50 exemples** sont sélectionnés dans le pool selon la stratégie choisie, ajoutés au set d'entraînement, et le modèle est réentraîné depuis zéro.

### Stratégies comparées

| Stratégie | Description |
|-----------|-------------|
| `random` | Sélection aléatoire dans le pool (baseline) |
| `uncertainty` | Exemples les plus incertains : score = 1 − \|2p − 1\|, où p = probabilité classe 1 |
| `kmeans` | Clustering KMeans sur embeddings CLS + exemple le plus incertain par cluster |


### Résultats — stratégie random

| Itération | Taille train | F1 | Accuracy |
|-----------|-------------|-----|----------|
| 0 | 1500 | 0.9735 | 0.9667 |
| 1 | 1550 | 0.9735 | 0.9667 |
| 2 | 1600 | 0.9735 | 0.9667 |
| 3 | 1650 | 0.9735 | 0.9667 |
| 4 | 1700 | 0.9735 | 0.9667 |
| 5 | 1750 | 0.9821 | 0.9778 |

### Résultats — stratégie uncertainty

| Itération | Taille train | F1 | Accuracy |
|-----------|-------------|-----|----------|
| 0 | 1500 | 0.9821 | 0.9778 |
| 1 | 1550 | 0.9730 | 0.9667 |
| 2 | 1600 | 0.9735 | 0.9667 |
| 3 | 1650 | 0.9735 | 0.9667 |
| 4 | 1700 | 0.9643 | 0.9556 |
| 5 | 1750 | 0.9735 | 0.9667 |

### Résultats — stratégie kmeans

| Itération | Taille train | F1 | Accuracy |
|-----------|-------------|-----|----------|
| 0 | 1500 | 0.9821 | 0.9778 |
| 1 | 1550 | 0.9735 | 0.9667 |
| 2 | 1600 | 0.9735 | 0.9667 |
| 3 | 1650 | 0.9739 | 0.9667 |
| 4 | 1700 | 0.9735 | 0.9667 |
| 5 | 1750 | 0.9649 | 0.9556 |

### Analyse

Le F1 oscille entre 0.964 et 0.982 sans tendance claire à la hausse. L'Active Learning n'apporte pas d'amélioration significative sur ce corpus pour deux raisons principales :

1. **Saturation du modèle** : CamemBERT-base atteint ses performances maximales dès l'itération 0 avec 1500 exemples annotés. Le dataset gold est suffisant pour cette tâche.
2. **Labels bruités dans le pool** : les labels automatiques du pool introduisent du bruit — des exemples mal étiquetés — qui limite ou annule le bénéfice de l'ajout de nouveaux exemples.

Ce résultat est en soi une conclusion scientifique : sur un corpus suffisamment annoté avec des labels gold, l'Active Learning avec un pool bruité ne justifie pas le surcoût.

---

## 8. Tokenisation et structures de données

### Avant tokenisation

Chaque exemple dans le Dataset HuggingFace contient :

| Champ | Type | Description |
|-------|------|-------------|
| `text` | `str` | Phrase reconstituée en joignant les tokens par un espace |
| `label` | `int` | 0 (pas de source) ou 1 (présence de source) |

### Après tokenisation

Le tokenizer CamemBERT (SentencePiece) découpe les textes en sous-mots et produit :

| Champ | Type | Description |
|-------|------|-------------|
| `input_ids` | `list[int]`, longueur max_length | Indices des tokens dans le vocabulaire |
| `attention_mask` | `list[int]`, longueur max_length | 1 = token réel, 0 = padding |
| `label` | `int` | Classe cible, inchangée |

Les séquences sont tronquées à 256 tokens et paddées à longueur fixe.

### Sorties du modèle

| Objet | Type | Description |
|-------|------|-------------|
| `logits` | `Tensor (batch, 2)` | Scores bruts avant softmax, un par classe |
| `probs` | `Tensor (batch, 2)` | Probabilités après softmax (somme = 1 par ligne) |
| `probs[:,1]` | `Tensor (batch,)` | Probabilité classe 1 (présence de source) |
| `y_pred` | `np.ndarray (N,)` | Labels prédits après seuillage à 0.75 |
| `y_true` | `np.ndarray (N,)` | Labels réels issus des annotations |

Pour l'Active Learning, les embeddings CLS (premier token de `last_hidden_state`) sont également extraits pour la stratégie KMeans.

---

## 9. API — Description des fonctions principales

### classification_binaire.py

`count_classes(dataset, name)` — affiche la distribution des classes 0 et 1 dans un split. `dataset` : liste de dicts avec clé `labels`. `name` : nom du split affiché.

`prepare_data(dataset)` → `(X, y)` — convertit une liste d'exemples en listes de phrases et de labels binaires. Retourne `X : list[str]` et `y : list[int]`.

`compute_metrics(eval_pred)` → `dict` — calcule accuracy et F1 à partir des logits et labels. Utilisé par le Trainer HuggingFace.

`WeightedTrainer` — sous-classe de `Trainer` qui remplace la loss standard par une Cross-Entropy pondérée (poids 1.0 / 1.5).

`tokenize(example)` → `dict` — tokenise une phrase avec padding à max_length et troncature.

### active_learning_final.py

`load_jsonl(path)` → `list[dict]` — charge un fichier JSONL ligne par ligne.

`prepare_data(dataset)` → `(X, y)` — même rôle que dans classification_binaire.py.

`build_hf_dataset(X, y, tokenizer, max_length)` → `Dataset` — construit un Dataset HuggingFace tokenisé.

`train_model(train_dataset, dev_dataset, config)` → `Trainer` — fine-tune CamemBERT et retourne le Trainer entraîné.

`get_embeddings(model, dataset)` → `np.ndarray` — extrait les embeddings CLS pour tous les exemples du pool.

`uncertainty_sampling(model, pool_dataset, k)` → `list[int]` — retourne les indices des k exemples les plus incertains.

`random_sampling(pool_data, k)` → `list[int]` — retourne k indices aléatoires.

`kmeans_sampling(model, pool_dataset, k)` → `list[int]` — clustering KMeans sur embeddings + exemple le plus incertain par cluster.

`run_active_learning(strategy, train_data, dev_data, test_data, pool_data, config)` — boucle AL principale : sélection → ajout → réentraînement → évaluation.

---

## 10. Gestion des exceptions

| Exception | Situation | Fichier |
|-----------|-----------|---------|
| `FileNotFoundError` | Fichier JSONL introuvable | classification_binaire.py, active_learning_final.py |
| `json.JSONDecodeError` | Ligne JSON invalide dans un fichier JSONL | classification_binaire.py |
| `RuntimeError` | Tokenizer ou modèle CamemBERT inaccessible (pas de connexion, nom incorrect) | classification_binaire.py |

---

## 11. Complexité algorithmique

Variables : N = nombre d'exemples, L = longueur moyenne des phrases (tokens), E = époques, H = hidden size CamemBERT (768), K = nombre de clusters KMeans, I = nombre d'itérations AL.

| Opération | Complexité |
|-----------|------------|
| Chargement des données | O(N) |
| `prepare_data` | O(N × L) |
| Tokenisation | O(N × max_length) |
| Entraînement | O(E × N × max_length × H) — dominé par les 12 couches transformeur |
| Évaluation | O(N × max_length × H) — forward sans gradient |
| Extraction embeddings CLS | O(N × max_length × H) |
| Uncertainty sampling | O(N) |
| KMeans sampling | O(N × K × H) |
| Boucle Active Learning complète | O(I × E × N × max_length × H) |

L'entraînement du transformeur domine toutes les autres étapes. Sur GPU T4 (Colab), une itération AL complète (3 époques, ~1500 exemples) prend environ 3-5 minutes.

---

## 12. Limites

**Labels bruités dans le pool** : les labels automatiques du pool sont imparfaits — un modèle entraîné sur 1500 exemples ne produit pas des annotations gold. Cela plafonne l'utilité de l'Active Learning.

**Saturation rapide** : CamemBERT-base atteint F1 ≈ 0.97 dès l'itération 0. Le corpus annoté est suffisant pour cette tâche binaire, ce qui laisse peu de marge d'amélioration.

**Reproducibilité** : les résultats varient légèrement d'un run à l'autre en raison de l'initialisation aléatoire du modèle.

---

## Contexte

Projet réalisé dans le cadre du M1 Linguistique Informatique — Université Grenoble Alpes (2025-2026).
Données issues du projet [DAPEPANE](https://gricad-gitlab.univ-grenoble-alpes.fr/m1-projet-s2/benjamin-nikita-lihui-sonia/dapepane).
