import openai

# Remplace par ta nouvelle clé API


try:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Bonjour, comment vas-tu ?"}]
    )
    print(response["choices"][0]["message"]["content"])
except openai.error.OpenAIError as e:
    print("Erreur :", e)
