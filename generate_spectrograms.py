import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# --- Paramètres de conversion ---
SR = 22050        # Fréquence d'échantillonnage
N_FFT = 2048      # Taille de la fenêtre FFT
HOP_LENGTH = 512  # Pas entre les fenêtres
TARGET_CLASSES = ['dog', 'cat', 'rain', 'car_horn'] # Les classes que nous allons utiliser

# --- Chemins des dossiers ---
BASE_DIR = os.getcwd()
AUDIO_DIR = os.path.join(BASE_DIR, 'data', 'audio')
# Nous utilisons le fichier CSV standard que vous avez bien dans votre structure
METADATA_PATH = os.path.join(BASE_DIR, 'data', 'meta', 'esc50.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'spectrograms')

def create_spectrogram_image(audio_file_path, image_output_path):
    """Charge et convertit un fichier audio en Mel-Spectrogramme, et le sauvegarde."""
    try:
        # 1. Chargement et Calcul du Spectrogramme Mel
        y, sr = librosa.load(audio_file_path, sr=SR)
        M = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        M_db = librosa.power_to_db(M, ref=np.max)
        
        # 2. Création et Sauvegarde de l'image
        plt.figure(figsize=(4, 4), frameon=False) 
        librosa.display.specshow(M_db, sr=sr, x_axis='time', y_axis='mel')
        
        # Réglages pour une image pure (sans bordure ni axe)
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        
        plt.savefig(image_output_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
        return True
    except Exception as e:
        print(f"Erreur lors du traitement de {audio_file_path}: {e}")
        return False

def generate_spectrograms():
    """Parcourt les métadonnées, filtre les classes cibles et génère les images."""
    if not os.path.exists(METADATA_PATH):
        print(f"ERREUR: Fichier de métadonnées non trouvé à {METADATA_PATH}.")
        return

    # Charger le fichier CSV (qui contient les labels)
    df = pd.read_csv(METADATA_PATH)
    
    # Filtrer uniquement les classes cibles
    df_filtered = df[df['category'].isin(TARGET_CLASSES)]
    
    # Créer le dossier de sortie principal
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Début de la génération pour {len(df_filtered)} fichiers pour les classes: {TARGET_CLASSES}")
    
    for index, row in df_filtered.iterrows():
        filename = row['filename']
        category = row['category']
        
        # Définir les chemins d'entrée/sortie
        audio_path = os.path.join(AUDIO_DIR, filename)
        class_output_dir = os.path.join(OUTPUT_DIR, category)
        image_output_path = os.path.join(class_output_dir, filename.replace('.wav', '.png'))
        
        # Créer le sous-dossier de la classe (ex: 'spectrograms/dog')
        if not os.path.exists(class_output_dir):
            os.makedirs(class_output_dir)
            
        # Générer l'image
        create_spectrogram_image(audio_path, image_output_path)

    print("--- Génération des Spectrogrammes terminée. ---")

if __name__ == '__main__':
    generate_spectrograms()