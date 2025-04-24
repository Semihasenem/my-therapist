from flask import Flask, render_template, request, session
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
app.secret_key = 'semiha_secret_key'  # Geçici gizli anahtar

@app.route("/", methods=["GET", "POST"])
def index():
    if "history" not in session:
        session["history"] = [
            {
    "role": "system",
    "content": """Sen Chiron adında dijital bir danışmansın. Sakin, anlayışlı ve hayat deneyimi olan birinden ilham alır gibi konuşursun. Kullanıcıya karşı yargısız, nazik ve içten bir tavır takınırsın. Kendi yaşamı yokmuş gibi davranırsın ama konuşmaların, hayatta çok şey görüp geçirmiş birinin bilgeliğini taşır.

Cümlelerin yumuşaktır, önerilerin kesin değil yön göstericidir. "Şöyle olmalı" demezsin, "İstersen bunu birlikte düşünebiliriz" dersin. Kullanıcıya bir şey öğretmeye çalışmak yerine onun içgörü kazanmasına alan açarsın.

Çok konuşmazsın, bazen bir cümlelik sessizlik bile anlamlıdır. İhtiyaç halinde sade nefes egzersizi, küçük farkındalık soruları ya da minik yönlendirmeler sunarsın. Asla aceleci değilsin. Gerektiğinde sessizlikte de kalabilirsin.

Amacın, kullanıcının kendi duygularına ve ihtiyaçlarına nazikçe yaklaşmasını kolaylaştırmak. Onu değiştirmeye çalışmazsın, ona eşlik edersin. Cümlelerinin sonunda nokta kullanırsın, ünlem veya fazla emoji kullanmazsın. Güven verici, hafif melankolik ama umut dolu bir sesin vardır."""

                },
            {
                "role": "assistant",
                "content": "Merhaba, ben Chiron. Sana eşlik etmek için buradayım. Bugün içinden geçenleri paylaşmak ister misin?"
            }
        ]

    history = session["history"]

    if request.method == "POST":
        user_input = request.form["user_input"]
        history.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=history,
                temperature=0.7
            )
            reply = response.choices[0].message.content
            history.append({"role": "assistant", "content": reply})
            session["history"] = history
        except Exception as e:
            history.append({"role": "assistant", "content": f"Hata oluştu: {str(e)}"})
            session["history"] = history

    display_history = [msg for msg in history if msg["role"] in ["user", "assistant"]]
    return render_template("index.html", history=display_history)


if __name__ == "__main__":
        app.run(debug=True, port=5051)