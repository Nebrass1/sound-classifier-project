import gradio as gr
import tensorflow as tf
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import json

# --- Paramètres du modèle (DOIVENT correspondre à ceux de l'entraînement) ---
MODEL_PATH = 'sound_classifier_model.h5'
LABELS_PATH = 'class_labels.json'
IMAGE_SIZE = (100, 100)
SR = 22050 
N_FFT = 2048
HOP_LENGTH = 512

# 1. Charger le Modèle et les Labels
model = None
label_map = {}

try:
    # Utilisation de safe_mode=False pour améliorer la compatibilité du chargement du format .h5
    model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False) 
    
    # Charger les labels depuis le JSON
    with open(LABELS_PATH, 'r') as f:
        class_indices = json.load(f)
        # On inverse pour avoir index: label (ex: 0: "dog")
        label_map = {v: k for k, v in class_indices.items()}
            
    print("Modèle et labels chargés avec succès.")
except Exception as e:
    # Ce message d'erreur sera affiché dans les logs du Space Hugging Face
    print(f"FATAL ERROR: Le modèle n'a PAS pu être chargé. Le fichier .h5 est-il présent ? Erreur détaillée: {e}") 
    model = None


def audio_to_spectrogram_for_prediction(audio_path):
    """
    Convertit un fichier audio en spectrogramme Mel, prêt pour la prédiction,
    et retourne l'image binaire pour l'affichage.
    """
    if not audio_path:
        return None, None
        
    y, sr = librosa.load(audio_path, sr=SR)
    
    # Gestion des clips audio très courts ou vides
    if len(y) == 0:
        return None, None
        
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    M_db = librosa.power_to_db(M, ref=np.max)
    
    # 1. Création de l'image Spectrogramme (pour l'affichage)
    plt.figure(figsize=(4, 4), frameon=False) 
    librosa.display.specshow(M_db, sr=sr, x_axis='time', y_axis='mel')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    
    # Sauvegarde dans la mémoire (buffer)
    spectrogram_buf = io.BytesIO()
    plt.savefig(spectrogram_buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()

    # 2. Traitement de l'image pour le modèle
    spectrogram_buf.seek(0) # Remet le curseur au début du buffer
    try:
        img = tf.keras.utils.load_img(spectrogram_buf, target_size=IMAGE_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0 # Normalisation
    except Exception as e:
        print(f"Erreur lors du traitement de l'image pour la prédiction: {e}")
        return None, None
    
    return spectrogram_buf, img_array 

def classify_sound(audio_input):
    """Fonction principale de classification appelée par Gradio."""
    # Afficher le message d'erreur si le chargement initial a échoué
    if model is None:
        return "**[Échec] Le modèle n'a pas pu être chargé au démarrage du Space.**", None

    if audio_input is None:
        return "Veuillez télécharger un fichier audio.", None
    
    spectrogram_buf, processed_image = audio_to_spectrogram_for_prediction(audio_input)
    
    if processed_image is None:
        return "Impossible de traiter l'audio ou fichier vide.", None

    # Prédiction
    predictions = model.predict(processed_image, verbose=0)[0]
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = label_map.get(predicted_class_index, "Inconnu")
    confidence = predictions[predicted_class_index] * 100
    
    # Formatage de la sortie
    result = f"## **Prédiction : {predicted_class_label.upper()}**\n\nConfiance : {confidence:.2f}%\n"
    
    return result, spectrogram_buf


# 2. Définition de l'Interface Gradio
iface = gr.Interface(
    fn=classify_sound, 
    inputs=gr.Audio(type="filepath", label="Téléchargez un fichier audio (.wav, .mp3, etc.)"),
    outputs=[
        gr.Markdown(label="Résultat de la Classification"), 
        gr.Image(type="bytes", label="Spectrogramme généré", width=400, height=400)
    ],
    title="Service Cloud de Classification de Sons (Transfer Learning)",
    description="Ce service utilise MobileNetV2 entraîné par Transfer Learning sur les spectrogrammes du dataset ESC-50 pour identifier le type de son.",
    examples=[] 
)

# 3. Lancement Local pour Test
if __name__ == "__main__":
    iface.launch()