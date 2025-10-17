import gradio as gr
import tensorflow as tf
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import time

# --- Paramètres du modèle (DOIVENT correspondre à ceux de l'entraînement) ---
MODEL_PATH = 'sound_classifier_model.h5'
IMAGE_SIZE = (100, 100)
SR = 22050 
N_FFT = 2048
HOP_LENGTH = 512

# 1. Charger le Modèle et les Labels
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Charger les labels (classes)
    label_map = {}
    with open('class_labels.txt', 'r') as f:
        for line in f:
            idx, label = line.strip().split(':')
            label_map[int(idx)] = label
            
    print("Modèle et labels chargés avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle/labels: {e}")
    # Utiliser un modèle bidon si le chargement échoue pour le test d'interface
    model = None


def audio_to_spectrogram(audio_path):
    """
    Convertit un fichier audio en spectrogramme Mel, prêt pour le modèle.
    """
    if not audio_path:
        return None
        
    y, sr = librosa.load(audio_path, sr=SR)
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    M_db = librosa.power_to_db(M, ref=np.max)
    
    # Créer l'image du spectrogramme en mémoire (pas sur disque)
    plt.figure(figsize=(4, 4), frameon=False) 
    librosa.display.specshow(M_db, sr=sr, x_axis='time', y_axis='mel')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()

    # Charger l'image depuis la mémoire (BytesIO) pour la prédicition Keras
    img = tf.keras.utils.load_img(buf, target_size=IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Ajout de la dimension du batch
    img_array /= 255.0 # Normalisation
    
    # Retourner l'image du spectrogramme pour l'affichage (BytesIO) et le tableau pour la prédiction
    return buf, img_array 

def classify_sound(audio_input):
    """
    Fonction principale de classification appelée par Gradio.
    """
    if model is None:
        return "Erreur: Modèle non chargé. Entraînez-le d'abord.", None

    if audio_input is None:
        return "Veuillez télécharger un fichier audio.", None
    
    start_time = time.time()
    
    # 1. Conversion
    spectrogram_buf, processed_image = audio_to_spectrogram(audio_input)
    
    # 2. Prédiction
    predictions = model.predict(processed_image, verbose=0)[0]
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = label_map.get(predicted_class_index, "Inconnu")
    confidence = predictions[predicted_class_index] * 100
    
    end_time = time.time()
    
    # 3. Formatage de la sortie
    result = f"**Prédiction : {predicted_class_label}**\nConfiance : {confidence:.2f}%\nTemps de traitement : {end_time - start_time:.2f}s"
    
    return result, spectrogram_buf


# 4. Définition de l'Interface Gradio
iface = gr.Interface(
    fn=classify_sound, 
    inputs=gr.Audio(type="filepath", label="Téléchargez un fichier audio (.wav, .mp3, etc.)"),
    outputs=[
        gr.Markdown(label="Résultat de la Classification"), 
        gr.Image(type="numpy", label="Spectrogramme généré", width=400, height=400)
    ],
    title="Service de Classification de Sons par Deep Learning",
    description="Utilise un modèle MobileNetV2 entraîné par Transfer Learning sur des spectrogrammes audio pour identifier le son (chien, chat, pluie, klaxon...).",
    examples=[
        # Vous pouvez ajouter des exemples de fichiers audio ici pour le test
    ]
)

if __name__ == "__main__":
    iface.launch()