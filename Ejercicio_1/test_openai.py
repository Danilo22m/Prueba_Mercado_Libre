import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    api_key = api_key.strip()
print(f"API Key encontrada: {'Si' if api_key else 'No'}")
print(f"API Key length: {len(api_key) if api_key else 0} caracteres")
print(f"API Key empieza con 'sk-': {api_key.startswith('sk-') if api_key else False}")

if api_key and len(api_key) > 20:
    try:
        client = OpenAI(api_key=api_key)
        print("\nCliente OpenAI inicializado correctamente")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Di hola en JSON: {\"saludo\": \"?\"}"}],
            temperature=0.0,
            max_tokens=50
        )

        print("\nRespuesta de OpenAI:")
        print(response.choices[0].message.content)
        print(f"\nTokens input: {response.usage.prompt_tokens}")
        print(f"Tokens output: {response.usage.completion_tokens}")
        print("\nTODO FUNCIONA CORRECTAMENTE!")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\nERROR: Necesitas configurar tu API key en el archivo .env")
    print("Debe ser una API key valida de OpenAI que empiece con 'sk-'")
