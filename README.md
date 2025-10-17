# 🎧 Sound Classifier Project: MobileNetV2 for Environmental Sound Classification

## 🚀 Projet Déployé

**Ceci est le code source du classificateur de sons déployé gratuitement sur Hugging Face Spaces.**

| Application en Ligne | Démo |
| :--- | :--- |
| [Lien vers votre Space Hugging Face](https://huggingface.co/spaces/Nebrass1/sound-classifier-nebrass) | **** |

## 🎯 Objectif du Projet

Ce projet vise à développer un classificateur capable d'identifier automatiquement des sons environnementaux (ex. : chien, pluie, klaxon) à partir de courts clips audio.

La solution utilise une approche de **Computer Vision** : les signaux audio sont convertis en **Spectrogrammes Mel**, transformant le problème d'analyse audio en un problème de classification d'images.

## 🧠 Architecture et Méthode

Nous avons utilisé une approche de **Transfer Learning** basée sur un modèle de classification d'images pré-entraîné, ce qui permet d'obtenir une bonne précision malgré un dataset d'entraînement réduit.

### Composants Clés

1.  **Pré-traitement Audio:** Utilisation de la librairie **Librosa** pour transformer les fichiers audio bruts en images de **spectrogrammes Mel**.
2.  **Modèle de Base:** **MobileNetV2** (pré-entraîné sur ImageNet). Ce modèle est idéal pour les images de petite taille comme les spectrogrammes.
3.  **Transfer Learning:** Le modèle est entraîné en deux phases (couches gelées, puis Fine-Tuning) pour affiner les poids de MobileNetV2 à la reconnaissance des caractéristiques visuelles spécifiques des spectrogrammes.

### Performance

Après l'entraînement complet (Transfer Learning et Fine-Tuning), le modèle a atteint une précision de validation (validation accuracy) de **87.50%**.

## 🛠️ Utilisation et Installation

### Prérequis

* Python 3.8+
* Git et Git LFS (Git Large File Storage)

### Structure du Projet
├── app.py # Script de l'interface Gradio et de la logique de prédiction.
├── train_model.py # Script d'entraînement MobileNetV2 (Transfer Learning).
├── requirements.txt # Dépendances Python.
├── sound_classifier_model.h5 # Modèle Keras entraîné (stocké via Git LFS).
└── class_labels.json # Mapping des classes (ex: {"dog": 0, "rain": 1, ...}).


### Démarrage Local de l'Application

Pour lancer l'interface web Gradio localement :

1.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Lancer l'Application :**
    ```bash
    python app.py
    ```
    *(Le modèle doit être présent dans le dossier racine.)*