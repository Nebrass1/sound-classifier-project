import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import os
import json # Pour sauvegarder l'index des classes plus facilement

# --- Paramètres ---
SPECTROGRAMS_DIR = os.path.join(os.getcwd(), 'spectrograms')
# La taille doit correspondre aux images générées
IMAGE_SIZE = (100, 100) 
BATCH_SIZE = 32
MODEL_OUTPUT_PATH = 'sound_classifier_model.h5'
LABELS_OUTPUT_PATH = 'class_labels.json'
EPOCHS_FT = 5 # Nombre d'époques pour le Fine-Tuning

def train_transfer_learning_model():
    # 1. Préparation des données avec Keras
    # ImageDataGenerator charge les images depuis les dossiers de classes
    datagen = ImageDataGenerator(
        rescale=1./255, # Normalisation des pixels entre 0 et 1 (nécessaire pour les modèles pré-entraînés)
        validation_split=0.2 # 20% des données pour la validation
    )

    # Gagner en robustesse: de légères augmentations d'images 
    # (cela peut améliorer la performance mais n'est pas obligatoire pour la première version)
    train_generator = datagen.flow_from_directory(
        SPECTROGRAMS_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical', 
        subset='training'
    )
    
    validation_generator = datagen.flow_from_directory(
        SPECTROGRAMS_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    NUM_CLASSES = train_generator.num_classes
    print(f"Nombre de classes détectées: {NUM_CLASSES}")
    
    # Sauvegarder l'association des labels (très important pour le déploiement)
    with open(LABELS_OUTPUT_PATH, 'w') as f:
        json.dump(train_generator.class_indices, f)
    print(f"Labels sauvegardés dans {LABELS_OUTPUT_PATH}")

    # 2. Construction du Modèle (Transfer Learning)
    
    # Charger MobileNetV2 (léger et efficace)
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, # On enlève la couche de classification d'ImageNet
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3) # Nos images sont 100x100 RGB
    )

    # Geler le modèle de base pour le Transfer Learning
    # Seules les nouvelles couches seront entraînées initialement
    base_model.trainable = False 

    # Ajout des nouvelles couches de classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Réduit les caractéristiques en un seul vecteur
    x = Dense(128, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x) # Couche de sortie pour nos N classes

    model = Model(inputs=base_model.input, outputs=predictions)

    # 3. Entraînement Initial (Transfer Learning)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("\n--- Début de l'Entraînement par Transfer Learning (couches gelées) ---")
    model.fit(
        train_generator,
        epochs=10, # Plus d'époques initiales pour trouver les bons poids de classification
        validation_data=validation_generator
    )

    # 4. Fine-Tuning (Ajustement des poids pour une meilleure performance)
    if EPOCHS_FT > 0:
        print("\n--- Début du Fine-Tuning (dé-gel et ajustement des couches supérieures) ---")
        
        # Dé-geler les dernières couches de MobileNet
        base_model.trainable = True
        
        # On ne dégèle que les dernières 40 couches pour ne pas corrompre les poids de base
        for layer in base_model.layers[:-40]: 
            layer.trainable = False

        # Recompiler le modèle avec un taux d'apprentissage très faible (pour ne pas perturber les poids)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), 
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        model.fit(
            train_generator,
            epochs=EPOCHS_FT,
            validation_data=validation_generator
        )

    # 5. Sauvegarde du modèle
    model.save(MODEL_OUTPUT_PATH)
    print(f"\nModèle sauvegardé avec succès: {MODEL_OUTPUT_PATH}")


if __name__ == '__main__':
    train_transfer_learning_model()