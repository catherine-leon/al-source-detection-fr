"""
Classification binaire de phrases — Détection de sources dans les questions parlementaires
Tâche : classifier chaque phrase comme contenant une référence à une source (classe 1)
ou non (classe 0), à partir des annotations IOB.

Modèle : CamemBERT-base fine-tuné avec Cross-Entropy Loss pondérée.

Utilisation :
    Adapter les chemins dans la section CHEMINS, puis exécuter le script.
    Les résultats (matrice de confusion, rapport de classification) sont affichés en sortie.

Dépendances :
    pip install torch transformers datasets scikit-learn sentencepiece accelerate
    pip install matplotlib seaborn pandas
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


# CHEMINS — à adapter selon votre environnement
TRAIN_PATH = '/content/train.json'
VALID_PATH = '/content/valid.json'
TEST_PATH  = '/content/test.json'


# HYPERPARAMÈTRES
CONFIG = {
    'model':         'camembert-base',
    'learning_rate': 2e-5,
    'batch_size':    16,
    'max_length':    256,
    'epochs':        3,
    'threshold':     0.75,  # seuil > 0.5 pour favoriser la précision sur la classe 1
}


"""
STRUCTURES DE DONNÉES
----------------------
1. Données brutes (liste de dicts)
   Chaque exemple : {"uid": str, "tokens": list[str], "labels": list[str]}
   Ex. : {"uid": "q123", "tokens": ["Le", "rapport"], "labels": ["O", "B-SRC"]}

2. Dataset HuggingFace (avant tokenisation)
   Dataset.from_dict({"text": list[str], "label": list[int]})
   "text"  : phrase reconstituée en joignant les tokens par un espace
   "label" : 0 (pas de source) ou 1 (présence de source)

3. Dataset HuggingFace (après tokenisation)
   Colonnes ajoutées par le tokenizer :
   - "input_ids"      : list[int] longueur max_length — indices des tokens dans le vocabulaire
   - "attention_mask" : list[int] longueur max_length — 1=token réel, 0=padding
   Format final : tenseurs PyTorch (via set_format("torch"))

4. Sorties du modèle
   - logits    : Tensor (batch_size, 2) — scores bruts avant softmax, un par classe
   - probs     : Tensor (batch_size, 2) — probabilités après softmax (somme = 1 par ligne)
   - probs[:,1]: Tensor (batch_size,)   — probabilité classe 1 (présence de source)
   - y_pred    : np.ndarray (N,)        — labels prédits après seuillage
   - y_true    : np.ndarray (N,)        — labels réels issus des annotations

COMPLEXITÉ
-----------
- Chargement des données  : O(N), N = nombre d'exemples
- prepare_data            : O(N × L), L = longueur moyenne des phrases en tokens
- Tokenisation (map)      : O(N × max_length)
- Entraînement            : O(E × N × max_length × H), E = epochs, H = hidden_size (768)
                            Dominé par les passes forward/backward (12 couches transformeur)
- Évaluation              : O(N × max_length × H) — passe forward sans gradient
- Matrice de confusion    : O(N) pour construire la matrice (2×2)
"""


def count_classes(dataset, name='dataset'):
    """
    Affiche la distribution des classes (0 et 1) dans un split du dataset.

    Args:
        dataset : liste de dicts avec clé 'labels'
        name    : nom du split affiché (ex. 'train', 'validation')
    """
    labels = []
    for item in dataset:
        label = 1 if any(l != 'O' for l in item['labels']) else 0
        labels.append(label)

    counter = Counter(labels)
    total   = len(labels)

    print(f'\n{name.upper()}')
    print('=' * 30)
    for cls in sorted(counter.keys()):
        count   = counter[cls]
        percent = count / total * 100
        print(f'  classe {cls} : {count} ({percent:.2f}%)')
    print(f'  total : {total}')


def prepare_data(dataset):
    """
    Convertit une liste d'exemples en listes (X, y) pour l'entraînement.

    Les tokens de chaque exemple sont joints en une phrase.
    Le label binaire vaut 1 si au moins un token porte une étiquette IOB différente de 'O'.

    Args:
        dataset : liste de dicts avec clés 'tokens' et 'labels'

    Returns:
        X : liste de phrases (str)
        y : liste de labels binaires (int)
    """
    X = []
    y = []
    for item in dataset:
        sentence = ' '.join(item['tokens'])
        label    = 1 if any(l != 'O' for l in item['labels']) else 0
        X.append(sentence)
        y.append(label)
    return X, y


def compute_metrics(eval_pred):
    """Calcule accuracy et F1 binaire à partir des logits et des labels (utilisé par Trainer)."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1':       f1_score(labels, preds),
    }


class WeightedTrainer(Trainer):
    """
    Trainer HuggingFace avec Cross-Entropy Loss pondérée.

    Poids : classe 0 = 1.0, classe 1 = 1.5.
    La pondération compense le déséquilibre en faveur de la classe positive.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels   = inputs.get('labels')
        outputs  = model(**inputs)
        logits   = outputs.get('logits')
        weights  = torch.tensor([1.0, 1.5]).to(logits.device)
        loss_fct = CrossEntropyLoss(weight=weights)
        loss     = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


if __name__ == '__main__':

    # Chargement des données
    train_data = []
    dev_data   = []
    test_data  = []

    for path, container in [(TRAIN_PATH, train_data), (VALID_PATH, dev_data), (TEST_PATH, test_data)]:
        try:
            with open(path, encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        container.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"[Avertissement] {path}, ligne {i+1} ignorée : {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Fichier introuvable : {path}\nVérifiez que le chemin est correct.")

    count_classes(train_data, 'train')
    count_classes(dev_data,   'validation')
    count_classes(test_data,  'test')

    # Préparation
    X_train, y_train = prepare_data(train_data)
    X_dev,   y_dev   = prepare_data(dev_data)
    X_test,  y_test  = prepare_data(test_data)

    train_dataset = Dataset.from_dict({'text': X_train, 'label': y_train})
    dev_dataset   = Dataset.from_dict({'text': X_dev,   'label': y_dev})
    test_dataset  = Dataset.from_dict({'text': X_test,  'label': y_test})

    try:
        tokenizer = CamembertTokenizer.from_pretrained(CONFIG['model'])
    except Exception as e:
        raise RuntimeError(f"Impossible de charger le tokenizer '{CONFIG['model']}' : {e}")

    def tokenize(example):
        """Tokenise une phrase avec padding fixe et troncature."""
        return tokenizer(
            example['text'],
            padding='max_length',
            truncation=True,
            max_length=CONFIG['max_length']
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    dev_dataset   = dev_dataset.map(tokenize, batched=True)
    test_dataset  = test_dataset.map(tokenize, batched=True)

    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    dev_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # Entraînement
    try:
        model = CamembertForSequenceClassification.from_pretrained(
            CONFIG['model'], num_labels=2
        )
    except Exception as e:
        raise RuntimeError(f"Impossible de charger le modèle '{CONFIG['model']}' : {e}")

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=CONFIG['epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        learning_rate=CONFIG['learning_rate'],
        save_strategy='no',
        logging_steps=100,
        report_to='none',
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Évaluation
    predictions = trainer.predict(test_dataset)

    logits = torch.tensor(predictions.predictions)
    probs  = F.softmax(logits, dim=1)

    threshold = CONFIG['threshold']
    y_pred    = (probs[:, 1] > threshold).int().numpy()
    y_true    = predictions.label_ids

    accuracy = accuracy_score(y_true, y_pred)
    f1       = f1_score(y_true, y_pred)

    print(f'\nSeuil de décision : {threshold}')
    print(f'Accuracy : {accuracy:.4f}')
    print(f'F1       : {f1:.4f}')
    print('\nRapport de classification :\n')
    print(classification_report(y_true, y_pred))

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.title('Matrice de confusion')
    plt.tight_layout()
    plt.show()
