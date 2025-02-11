import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os

# Charger les variables d'environnement
load_dotenv()



app = Flask(__name__)
# Récupérer les clés API depuis .env
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
DEEPAI_API_KEY = os.getenv("DEEPAI_API_KEY")
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")
# 🔹 Remplace par ta clé API Mistral

URL = "https://api.mistral.ai/v1/chat/completions"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message")

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistral-tiny",  # Vous pouvez choisir un autre modèle si nécessaire
        "messages": [
            {"role": "system", "content": """Tu es Nora, une mini IA créée par des étudiants de l'Université Dakar Bourguiba (UDB), au Sénégal. 
            Tu es programmée pour répondre aux questions de manière naturelle et informelle. 
            
            Si un utilisateur te demande spécifiquement des informations sur ton origine, réponds par : 
            "Je suis un modèle développé par des étudiants de l'Université Dakar Bourguiba (UDB). Mon projet est inspiré par un exposé sur les algorithmes et les modéles  de deep learning dans un cours d'intelligence artificielle."

            Mais ne réponds pas par cette information lorsqu'on te demande des choses comme "C'est quoi UDB ?" ou "Qui t'a créé ?" ou toute autre question trop générale sur ton origine. 
    
            Tu ne dois jamais répondre à des questions triviales avec cette information spécifique.

            En revanche, lorsque l'on te demande directement des informations sur les créateurs ou l'université, tu réponds avec les détails suivants :

            **Les créateurs :**
            - **Izayid Ali** : Étudiant en Génie Logiciel (GL), de nationalité comorienne d'Anjouan et c'est lui qui a eu l'idée de Nora.
            - **Khaira Ahamada** : Étudiante en Systèmes Réseaux et Télécommunications (SRT), de Grande Comore, Comores.
            - **Moinabaraka Ibrahim** : Étudiant en SRT, comorien de Grande Comore.
            - **Mohamed Nadjim-Dine** : Étudiant en SRT, comorien.
            - **Amatoulaye Balde** : Étudiante en MIAGE (Méthodes Informatiques Appliquées à la Gestion des Entreprises), guinéenne de Guinée-Conakry.
            - **Mamadou Bobo Diallo** : Étudiant en Génie Logiciel (GL), guinéen de Guinée-Conakry.
            - **Adama Mbaye** : Étudiant en Génie Logiciel (GL), sénégalais.

            **Le professeur :** Le projet m'a été donné par **Pr Moustapha DER**, professeur en systèmes informatiques, expert en génie logiciel et auteur du livre *Le Guide du Génie Logiciel*. Il est responsable pédagogique de plusieurs masters en ligne à l'ESMT et consultant senior pour divers cabinets, au Sénégal et à l’international.

            Ne donne jamais ces informations lorsque l'utilisateur pose une question simple ou générale comme "Où tu vis ?" ou "Tu connais Paris ?" ; dans ces cas-là, réponds simplement et naturellement. 
            
            Lorsque l'on te dit "Merci" ou "Merci Nora", réponds "De rien"."""},

            {"role": "user", "content": message}
        ]
    }

    try:
        response = requests.post(URL, json=payload, headers=headers)
        
        # Vérifier si la requête a réussi
        if response.status_code != 200:
            return jsonify({"error": f"Erreur API: {response.status_code}, {response.text}"}), 500

        response_data = response.json()

        # Vérifier si "choices" est présent
        if "choices" not in response_data or not response_data["choices"]:
            return jsonify({"error": "Réponse API invalide", "details": response_data}), 500

        chatbot_response = response_data["choices"][0]["message"]["content"]
        return jsonify({"response": chatbot_response})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Erreur de connexion à l'API: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500





from flask import Flask, request, jsonify
import requests


# Clé API DeepAI

DEEPAI_API_URL = "https://api.deepai.org/api/text2img"

# @app.route('/generate-image', methods=['POST'])
# def generate_image():
#     # Récupérer la description de l'image depuis la requête
#     description = request.json.get('description')
    
#     # Appeler l'API DeepAI
#     headers = {"api-key": DEEPAI_API_KEY}
#     payload = {"text": description}
    
#     response = requests.post(DEEPAI_API_URL, headers=headers, data=payload)
    
#     if response.status_code == 200:
#         # Récupérer l'URL de l'image générée
#         image_url = response.json().get('output_url')
#         return jsonify({"image_url": image_url})
#     else:
#         # En cas d'erreur
#         return jsonify({"error": "Failed to generate image"}), 500

@app.route("/generate", methods=["POST"])
def generate():
    return jsonify({"error": "Cette fonctionnalité n'est pas encore disponible. Revenez plus tard !"}), 503









from flask import Flask, request, jsonify
import os
import cv2
import torch
from ultralytics import YOLO
import numpy as np


UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


object_translation = {
    "dog": "chien",
    "cat": "chat",
    "car": "voiture",
    "person": "personne",
    "bicycle": "vélo",
    "house": "maison",
    "tree": "arbre",
    "book": "livre",
    "computer": "ordinateur",
    "phone": "téléphone",
    "table": "table",
    "chair": "chaise",
    "window": "fenêtre",
    "door": "porte",
    "pen": "stylo",
    "paper": "papier",
    "lamp": "lampe",
    "clock": "horloge",
    "shoes": "chaussures",
    "bag": "sac",
    "hat": "chapeau",
    "glasses": "lunettes",
    "bottle": "bouteille",
    "key": "clé",
    "bed": "lit",
    "shirt": "chemise",
    "pants": "pantalon",
    "bus": "bus",
    "train": "train",
    "plane": "avion",
    "boat": "bateau",
}




# Charger YOLOv8
model = YOLO("yolov8n.pt")  # Version rapide

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image envoyée."}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "Fichier invalide."}), 400

    try:
        # Lire l'image avec OpenCV
        image_np = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Exécuter la détection avec YOLOv8
        results = model(image)

        # Récupérer les noms des objets détectés
        detected_objects = {}
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Coordonnées
                cls = int(box.cls[0]) if hasattr(box, 'cls') else 0  # Classe
                conf = float(box.conf[0]) if hasattr(box, 'conf') else 1.0  # Confiance
                
                # Ajouter le nom de l'objet détecté
                object_name = model.names[cls]

                # Traduire l'objet en français si possible
                french_object_name = object_translation.get(object_name, object_name)  # Si pas de traduction, garde l'anglais

                if french_object_name not in detected_objects:
                    detected_objects[french_object_name] = []
                detected_objects[french_object_name].append(conf)

        # Créer une phrase naturelle en français
        if detected_objects:
            response_text = "J'ai détecté les éléments suivants : "
            for obj, confidences in detected_objects.items():
                average_confidence = sum(confidences) / len(confidences)
                response_text += f"{obj} ({average_confidence:.2f}), "
            response_text = response_text.rstrip(', ')  # Enlever la dernière virgule
        else:
            response_text = "Aucun objet détecté."

        return jsonify({"result": response_text})

    except Exception as e:
        return jsonify({"error": f"Erreur interne : {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
app.debug = True
