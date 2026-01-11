import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
print(f"API Key encontrada: {'Si' if api_key else 'No'}")
print(f"API Key length: {len(api_key) if api_key else 0} caracteres")
print(f"API Key empieza con 'gsk_': {api_key.startswith('gsk_') if api_key else False}")

if api_key and len(api_key) > 20:
    try:
        client = Groq(api_key=api_key)
        print("\nCliente Groq inicializado correctamente")

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Di hola en JSON: {\"saludo\": \"?\"}"}],
            temperature=0.0,
            max_tokens=50
        )

        print("\nRespuesta de Groq:")
        print(response.choices[0].message.content)
        print("\nTODO FUNCIONA CORRECTAMENTE!")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\nERROR: Necesitas configurar tu API key en el archivo .env")
    print("Debe ser una API key valida de Groq que empiece con 'gsk_'")
