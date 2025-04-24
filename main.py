import os
from openai import OpenAI
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# OpenAI istemcisi
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat_with_therapist():
    print("AI Terapist'e hoş geldiniz. Çıkmak için 'çık' yazın.")

    messages = [
        {"role": "system", "content": "Sen empatik ve destekleyici bir psikoterapistsin. Kullanıcılara içgörü kazandırmaya çalış, yargılayıcı olma."}
    ]

    while True:
        user_input = input("Sen: ")
        if user_input.lower() in ["çık", "exit", "quit"]:
            print("Görüşmek üzere 🧠✨")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7
            )

            reply = response.choices[0].message.content
            print("Terapist:", reply)

            messages.append({"role": "assistant", "content": reply})

        except Exception as e:
            print("Bir hata oluştu:", e)

if __name__ == "__main__":
    chat_with_therapist()