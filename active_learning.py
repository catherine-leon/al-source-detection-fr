"""
Active Learning pour la détection de sources dans les questions parlementaires

Tâche : classification binaire de phrases (contient une référence à une source ou non).
Modèle : CamemBERT-base fine-tuné (classification de séquences).

Trois stratégies d'Active Learning disponibles :
    - random      : sélection aléatoire (baseline)
    - uncertainty : sélection des exemples les plus incertains pour le modèle
    - kmeans      : clustering sémantique + sélection de l'exemple le plus incertain par cluster

Utilisation :
    python3 active_learning_final.py --strategy random
    python3 active_learning_final.py --strategy uncertainty
    python3 active_learning_final.py --strategy kmeans

Les résultats de chaque stratégie sont sauvegardés après chaque itération dans :
    <BASE_DIR>/results_<strategy>.json


CONFIGURATION REQUISE

1. Installer les dépendances :
       pip install torch transformers datasets scikit-learn sentencepiece accelerate

2. Adapter BASE_DIR (ligne ~50) au répertoire contenant les fichiers de données.

3. Placer les fichiers suivants dans BASE_DIR :
       - train_augmented.json          (split d'entraînement, format JSONL)
       - valid_augmented.json          (split de validation, format JSONL)
       - test_augmented.json           (split de test, format JSONL)
       - predictions_with_context.json (pool non annoté, format JSON liste)

Format attendu pour chaque exemple dans les fichiers JSONL :
    {"uid": "...", "tokens": ["mot1", "mot2", ...], "labels": ["O", "B-SRC", ...]}

Format attendu pour le pool (predictions_with_context.json) :
    [{"uid": "...", "tokens": [...], "labels": [...]}, ...]
    Les labels du pool sont issus de prédictions automatiques (peuvent être bruités).


ARCHITECTURE

- WeightedTrainer       : Trainer HuggingFace avec poids de classe (1.0 pour classe 0, 1.5 pour classe 1)
                          pour compenser le déséquilibre résiduel dans les données.
- random_sampling       : sélection aléatoire de k exemples dans le pool.
- model_uncertainty     : sélection des k exemples avec le score d'incertitude le plus élevé.
                          Score = 1 - |2p - 1| où p = P(classe=1). Vaut 1 si p=0.5, 0 si p=0 ou 1.
- kmeans_semantic       : clustering KMeans (k clusters) sur les embeddings CLS du pool,
                          puis sélection de l'exemple le plus incertain dans chaque cluster.
                          Le clustering est calculé une seule fois (itération 0), puis mis à jour
                          en supprimant les exemples sélectionnés.
- run_al_loop           : boucle principale AL — entraîne le modèle, évalue sur le test,
                          sélectionne de nouveaux exemples, répète.
"""

import json
import random
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from datasets import Dataset
from transformers import (
    CamembertTokenizer,
    CamembertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, accuracy_score



# CHEMINS — à adapter selon votre environnement

BASE_DIR = '/path/to/your/data'  # à adapter selon votre environnement

TRAIN_PATH = f'{BASE_DIR}/train_augmented.json'
VALID_PATH = f'{BASE_DIR}/valid_augmented.json'
TEST_PATH  = f'{BASE_DIR}/test_augmented.json'
POOL_PATH  = f'{BASE_DIR}/predictions_with_context.json'



# HYPERPARAMÈTRES

CONFIG = {
    "model_name":    "camembert-base",  # modèle HuggingFace (téléchargé automatiquement)
    "learning_rate": 2e-5,
    "batch_size":    16,
    "max_length":    256,               # longueur max en tokens (tronqué si dépassé)
    "epochs":        3,                 # epochs par itération AL
    "al_k":          50,                # nombre d'exemples sélectionnés par itération
    "al_iterations": 10,                 # nombre d'itérations AL (hors itération 0)
    "seed":          42,
}



# CHARGEMENT ET PRÉPARATION DES DONNÉES


def load_jsonl(path):
    """Charge un fichier JSONL (une ligne = un exemple JSON) et retourne une liste de dicts."""
    data = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def item_label(item):
    """
    Détermine le label binaire d'un exemple à partir de ses étiquettes IOB.
    Retourne 1 si au moins un token porte un label différent de 'O' (= présence de source),
    retourne 0 sinon.
    """
    return 1 if any(l != 'O' for l in item['labels']) else 0


def prepare_hf_dataset(items, tokenizer, max_length):
    """
    Convertit une liste d'exemples (format interne) en Dataset HuggingFace tokenisé.

    Les tokens de chaque exemple sont joints en une chaîne de caractères,
    puis tokenisés par CamemBERT avec padding et troncature.

    Args:
        items      : liste de dicts avec clés 'tokens' et 'labels'
        tokenizer  : CamembertTokenizer
        max_length : longueur maximale en tokens

    Returns:
        Dataset HuggingFace avec colonnes input_ids, attention_mask, label (format torch)
    """
    X  = [' '.join(item['tokens']) for item in items]
    y  = [item_label(item) for item in items]
    ds = Dataset.from_dict({'text': X, 'label': y})

    def tokenize(ex):
        return tokenizer(
            ex['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )

    ds = ds.map(tokenize, batched=True)
    ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    return ds


def compute_metrics(eval_pred):
    """Calcule accuracy, F1 binaire et F1 macro à partir des logits et des labels."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1':       f1_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro'),
    }


# MODÈLE

class WeightedTrainer(Trainer):
    """
    Trainer HuggingFace avec Cross-Entropy Loss pondérée.

    Poids : classe 0 (pas de source) = 1.0, classe 1 (source) = 1.5.
    La pondération compense le déséquilibre résiduel dans les données (63% positifs).
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.get('labels')
        outputs = model(**inputs)
        logits  = outputs.get('logits')
        weights = torch.tensor([1.0, 1.5]).to(logits.device)
        loss    = CrossEntropyLoss(weight=weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train_model(train_items, dev_items, tokenizer, config):
    """
    Initialise et entraîne un nouveau CamemBERT pour la classification binaire.

    Un nouveau modèle est instancié à chaque appel (pas de warm-start entre itérations AL).
    Les checkpoints intermédiaires ne sont pas sauvegardés (save_strategy='no')
    pour économiser de l'espace disque.

    Args:
        train_items : exemples d'entraînement (liste de dicts)
        dev_items   : exemples de validation (liste de dicts)
        tokenizer   : CamembertTokenizer
        config      : dict de configuration (CONFIG)

    Returns:
        (model, trainer) : modèle entraîné et Trainer HuggingFace associé
    """
    model    = CamembertForSequenceClassification.from_pretrained(
        config['model_name'], num_labels=2
    )
    train_ds = prepare_hf_dataset(train_items, tokenizer, config['max_length'])
    dev_ds   = prepare_hf_dataset(dev_items,   tokenizer, config['max_length'])

    args = TrainingArguments(
        output_dir='./al_checkpoints',
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        save_strategy='no',
        logging_steps=100,
        report_to='none',
        seed=config['seed'],
    )
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return model, trainer


def evaluate_on_test(trainer, test_items, tokenizer, config):
    """
    Évalue le modèle sur le split de test et retourne F1, F1 macro et accuracy.

    Args:
        trainer    : WeightedTrainer après entraînement
        test_items : exemples de test (liste de dicts)
        tokenizer  : CamembertTokenizer
        config     : dict de configuration

    Returns:
        dict avec clés 'f1', 'f1_macro', 'accuracy'
    """
    test_ds = prepare_hf_dataset(test_items, tokenizer, config['max_length'])
    preds   = trainer.predict(test_ds)
    y_pred  = np.argmax(preds.predictions, axis=1)
    y_true  = preds.label_ids
    return {
        'f1':       f1_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'accuracy': accuracy_score(y_true, y_pred),
    }


# STRATÉGIES D'ACTIVE LEARNING

@torch.no_grad()
def get_cls_embeddings(items, tokenizer, model, config, batch_size=64):
    """
    Extrait les embeddings du token [CLS] pour chaque exemple du pool.

    Le token [CLS] (position 0 dans last_hidden_state) représente la phrase entière
    après encodage par CamemBERT. Utilisé pour le clustering KMeans.

    Args:
        items      : liste d'exemples (dicts avec clé 'tokens')
        tokenizer  : CamembertTokenizer
        model      : CamemBERT entraîné
        config     : dict de configuration
        batch_size : taille des batchs pour l'inférence

    Returns:
        np.ndarray de forme (N, hidden_size) — un vecteur par exemple
    """
    model.eval()
    device   = next(model.parameters()).device
    all_embs = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        texts = [' '.join(item['tokens']) for item in batch]
        enc   = tokenizer(texts, padding='max_length', truncation=True,
                          max_length=config['max_length'], return_tensors='pt')
        enc   = {k: v.to(device) for k, v in enc.items()}
        out   = model.roberta(**enc)
        cls   = out.last_hidden_state[:, 0, :]
        all_embs.append(cls.cpu().numpy())
    return np.vstack(all_embs)


@torch.no_grad()
def get_uncertainty_scores(items, tokenizer, model, config, batch_size=64):
    """
    Calcule le score d'incertitude pour chaque exemple du pool.

    Score = 1 - |2p - 1|  où p = P(classe=1 | texte).
    - Score = 1  si p = 0.5 (modèle totalement incertain)
    - Score = 0  si p = 0 ou p = 1 (modèle très confiant)

    Args:
        items      : liste d'exemples (dicts avec clé 'tokens')
        tokenizer  : CamembertTokenizer
        model      : CamemBERT entraîné
        config     : dict de configuration
        batch_size : taille des batchs pour l'inférence

    Returns:
        np.ndarray de forme (N,) — un score par exemple, entre 0 et 1
    """
    model.eval()
    device = next(model.parameters()).device
    all_p1 = []
    for i in range(0, len(items), batch_size):
        batch  = items[i : i + batch_size]
        texts  = [' '.join(item['tokens']) for item in batch]
        enc    = tokenizer(texts, padding='max_length', truncation=True,
                           max_length=config['max_length'], return_tensors='pt')
        enc    = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_p1.extend(probs)
    p = np.array(all_p1)
    return 1.0 - np.abs(2 * p - 1)


def model_uncertainty(pool_items, tokenizer, model, config, k):
    """
    Sélectionne les k exemples du pool avec le score d'incertitude le plus élevé.

    Args:
        pool_items : liste d'exemples non encore annotés
        tokenizer  : CamembertTokenizer
        model      : CamemBERT entraîné
        config     : dict de configuration
        k          : nombre d'exemples à sélectionner

    Returns:
        (selected, remaining) : exemples choisis et pool résiduel
    """
    scores    = get_uncertainty_scores(pool_items, tokenizer, model, config)
    top_idx   = set(np.argsort(scores)[-k:].tolist())
    selected  = [item for i, item in enumerate(pool_items) if i in top_idx]
    remaining = [item for i, item in enumerate(pool_items) if i not in top_idx]
    return selected, remaining


def kmeans_semantic(pool_items, tokenizer, model, config, k, cluster_labels=None):
    """
    Sélectionne k exemples en combinant clustering sémantique et incertitude.

    Algorithme :
        1. Calculer les scores d'incertitude pour tout le pool.
        2. Si cluster_labels est None, calculer les embeddings CLS et appliquer KMeans(k clusters).
        3. Pour chaque cluster, sélectionner l'exemple le plus incertain.
        4. Retourner les exemples sélectionnés et mettre à jour les labels de cluster
           (en supprimant les indices sélectionnés).

    Le clustering est calculé une seule fois (à l'itération 0), puis réutilisé
    en retirant les exemples déjà sélectionnés — évite de recalculer les embeddings
    à chaque itération.

    Args:
        pool_items     : liste d'exemples non encore annotés
        tokenizer      : CamembertTokenizer
        model          : CamemBERT entraîné
        config         : dict de configuration
        k              : nombre de clusters = nombre d'exemples à sélectionner
        cluster_labels : labels de cluster existants (np.ndarray ou None)

    Returns:
        (selected, remaining, new_cluster_labels)
    """
    scores = get_uncertainty_scores(pool_items, tokenizer, model, config)
    if cluster_labels is None:
        print("  Computing KMeans clusters...")
        embeddings     = get_cls_embeddings(pool_items, tokenizer, model, config)
        kmeans         = KMeans(n_clusters=k, random_state=config['seed'], n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

    selected_idx = []
    for c in range(k):
        c_idx = np.where(cluster_labels == c)[0]
        if len(c_idx) == 0:
            continue
        best = c_idx[np.argmax(scores[c_idx])]
        selected_idx.append(int(best))

    keep_mask            = np.ones(len(pool_items), dtype=bool)
    keep_mask[selected_idx] = False
    selected             = [pool_items[i] for i in selected_idx]
    remaining            = [item for i, item in enumerate(pool_items) if keep_mask[i]]
    new_cluster_lbl      = cluster_labels[keep_mask]
    return selected, remaining, new_cluster_lbl


def random_sampling(pool_items, k, seed):
    """
    Sélectionne k exemples aléatoirement dans le pool (baseline).

    Args:
        pool_items : liste d'exemples non encore annotés
        k          : nombre d'exemples à sélectionner
        seed       : graine aléatoire (varie à chaque itération pour éviter les répétitions)

    Returns:
        (selected, remaining) : exemples choisis et pool résiduel
    """
    rng       = random.Random(seed)
    selected  = rng.sample(pool_items, k)
    sel_set   = {id(s) for s in selected}
    remaining = [item for item in pool_items if id(item) not in sel_set]
    return selected, remaining

# BOUCLE PRINCIPALE D'ACTIVE LEARNING

def run_al_loop(strategy_name, train_data, dev_data, test_data, pool, tokenizer, config):
    """
    Exécute la boucle Active Learning pour une stratégie donnée.

    À chaque itération :
        1. Sélectionner k exemples dans le pool selon la stratégie choisie.
        2. Les ajouter au set d'entraînement.
        3. Réentraîner le modèle depuis zéro sur le set d'entraînement élargi.
        4. Évaluer sur le test et sauvegarder les résultats.

    L'itération 0 correspond à l'entraînement sur le dataset initial (sans AL).
    Les résultats sont sauvegardés après chaque itération dans results_<strategy>.json
    (permettant de récupérer les résultats partiels en cas d'interruption).

    Args:
        strategy_name : 'random', 'uncertainty' ou 'kmeans'
        train_data    : liste d'exemples d'entraînement initiaux
        dev_data      : liste d'exemples de validation
        test_data     : liste d'exemples de test
        pool          : liste d'exemples non annotés (filtrés — sans doublons avec le dataset)
        tokenizer     : CamembertTokenizer
        config        : dict de configuration (CONFIG)

    Returns:
        list de dicts : résultats par itération (f1, f1_macro, accuracy, iteration, train_size)
    """
    results        = []
    current_train  = list(train_data)
    current_pool   = list(pool)
    cluster_labels = None

    print(f"\n{'='*60}")
    print(f"Strategy: {strategy_name.upper()}")
    print(f"{'='*60}")

    print(f"\n[Iter 0] Training on {len(current_train)} examples...")
    model, trainer = train_model(current_train, dev_data, tokenizer, config)
    metrics = evaluate_on_test(trainer, test_data, tokenizer, config)
    metrics.update({'iteration': 0, 'train_size': len(current_train)})
    results.append(metrics)
    print(f"  -> F1={metrics['f1']:.4f}  F1-macro={metrics['f1_macro']:.4f}")

    with open(f'{BASE_DIR}/results_{strategy_name}.json', 'w') as f:
        json.dump(results, f, indent=2)

    if strategy_name == 'kmeans':
        print(f"\n[KMeans init] Clustering {len(current_pool)} pool examples...")
        embeddings     = get_cls_embeddings(current_pool, tokenizer, model, config)
        kmeans         = KMeans(n_clusters=config['al_k'],
                                random_state=config['seed'], n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        print(f"  -> {config['al_k']} clusters created.")

    for it in range(1, config['al_iterations'] + 1):
        if len(current_pool) < config['al_k']:
            print("Pool exhausted.")
            break

        print(f"\n[Iter {it}] Selecting {config['al_k']} examples ({strategy_name})...")

        if strategy_name == 'random':
            selected, current_pool = random_sampling(
                current_pool, config['al_k'], seed=config['seed'] + it)
        elif strategy_name == 'uncertainty':
            selected, current_pool = model_uncertainty(
                current_pool, tokenizer, model, config, config['al_k'])
        elif strategy_name == 'kmeans':
            selected, current_pool, cluster_labels = kmeans_semantic(
                current_pool, tokenizer, model, config,
                config['al_k'], cluster_labels)

        current_train = current_train + selected
        print(f"  Train: {len(current_train)} | Pool: {len(current_pool)}")

        model, trainer = train_model(current_train, dev_data, tokenizer, config)
        metrics = evaluate_on_test(trainer, test_data, tokenizer, config)
        metrics.update({'iteration': it, 'train_size': len(current_train)})
        results.append(metrics)
        print(f"  -> F1={metrics['f1']:.4f}  F1-macro={metrics['f1_macro']:.4f}")

        with open(f'{BASE_DIR}/results_{strategy_name}.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to results_{strategy_name}.json")

    return results



# POINT D'ENTRÉE

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Active Learning pour la détection de sources (questions parlementaires).'
    )
    parser.add_argument(
        '--strategy', type=str, required=True,
        choices=['random', 'uncertainty', 'kmeans'],
        help="Stratégie AL : 'random' (baseline), 'uncertainty' (incertitude), 'kmeans' (clustering sémantique)"
    )
    args = parser.parse_args()

    print("Loading data...")
    train_data = load_jsonl(TRAIN_PATH)
    dev_data   = load_jsonl(VALID_PATH)
    test_data  = load_jsonl(TEST_PATH)

    with open(POOL_PATH, encoding='utf-8') as f:
        pool_raw = json.load(f)

    # Exclure du pool les exemples déjà présents dans le dataset annoté (déduplication par uid)
    used_uids = {item['uid'] for item in train_data + dev_data + test_data}
    pool = [item for item in pool_raw if item['uid'] not in used_uids]
    print(f"Pool after dedup: {len(pool)} (removed {len(pool_raw) - len(pool)})")

    tokenizer = CamembertTokenizer.from_pretrained(CONFIG['model_name'])

    results = run_al_loop(
        args.strategy, train_data, dev_data, test_data,
        pool, tokenizer, CONFIG
    )

    print(f"\nDone. Results saved to {BASE_DIR}/results_{args.strategy}.json")
