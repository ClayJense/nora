function sendChat() {
    // Récupérer le message entré par l'utilisateur
    var message = document.getElementById("chatInput").value;

    // Si le message est vide, ne rien faire
    if (!message) return;

    // Envoyer une requête AJAX à Flask
    fetch("/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: message })  // Passer le message en JSON
    })
    .then(response => response.json())
    .then(data => {
        // Afficher la réponse de l'IA dans la section de chat
        if (data.response) {
            document.getElementById("chatOutput").innerHTML = data.response;
        } else if (data.error) {
            document.getElementById("chatOutput").innerHTML = "Erreur: " + data.error;
        }
    })
    .catch(error => {
        document.getElementById("chatOutput").innerHTML = "Une erreur est survenue.";
    });
}



function generateImage() {
    // Récupérer la description de l'image
    const description = document.getElementById('imageGenInput').value;

    // Envoyer la description au backend
    fetch('/generate-image', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ description: description }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.image_url) {
            // Afficher l'image générée
            const imgElement = document.getElementById('generatedImage');
            imgElement.src = data.image_url;
            imgElement.style.display = 'block';
        } else {
            alert("Cette fonctionnalité n'est pas encore disponible. Revenez plus tard !");
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}




function analyzeImage() {
    const fileInput = document.getElementById("imageAnalysisInput");
    const analysisResult = document.getElementById("analysisResult");

    const formData = new FormData();
    formData.append("image", fileInput.files[0]);

    fetch("https://nora-1hry.onrender.com/detect", {  // Remplace par ton URL Render
        method: "POST",
        body: formData,
    })
    .then(response => response.json())  // Assure-toi que la réponse est au format JSON
    .then(data => {
        if (data.result) {
            analysisResult.textContent = `Objets détectés : ${data.result}`;
        } else {
            analysisResult.textContent = "Aucun objet détecté.";
        }
    })
    .catch(error => {
        console.error("Erreur:", error);
        analysisResult.textContent = "Une erreur est survenue lors de l'analyse de l'image.";
    });
}






