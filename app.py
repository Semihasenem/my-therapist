from flask import Flask, render_template, request, session
import os
from openai import OpenAI
from dotenv import load_dotenv
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import random
from flask import redirect, url_for

with open("cbt_templates.json", "r", encoding="utf-8") as f:
    cbt_templates = json.load(f)
# ENV yükle
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# OpenAI API client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# HuggingFace model
emotion_classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/bert-base-go-emotion",
    top_k=1
)
def get_username():
    isim = session.get("profile", {}).get("isim", "").strip()
    # Kullanıcı boş bıraktıysa ya da anonim kalmak istediğini belirttiyse
    if not isim or isim.lower() in ["istemiyorum", "yok", "hayır", "boş", "-"]:
        return None
    return isim
def personalize_response(response):
    username = get_username()
    # Kullanıcı ismi varsa ve %33 olasılıkla kişiselleştir
    if username and random.random() < True:
        secenek = random.choice([
            f"{username}, {response}",
            f"{response} Ne dersin, {username}?",
            f"{response}"
        ])
        return secenek
    return response
def detect_emotion(text):
    try:
        result = emotion_classifier(text)
        # Sonuç beklenen gibi geldiyse...
        if result and isinstance(result[0], list) and len(result[0]) > 0 and 'label' in result[0][0]:
            return result[0][0]['label'].lower()
        # Alternatif bazı modellerde dict olabilir
        elif result and isinstance(result[0], dict) and 'label' in result[0]:
            return result[0]['label'].lower()
        else:
            return "neutral"  # Hiç sonuç yoksa nötr döndür
    except Exception as e:
        # print(f"Duygu analizi hatası: {e}")  # Debug için açabilirsin
        return "neutral"

emotion_map = {
    "sadness": "üzgün",
    "joy": "mutlu",
    "anger": "öfke",
    "fear": "korku",
    "loneliness": "yalnız",
    "disappointment": "hayal kırıklığı",
    "anxiety": "kaygı",
    "neutral": "nötr"
}

cbt_responses_tr = {
    "kaygı": "Kaygının sana anlatmaya çalıştığı bir şey olabilir mi? Belki biraz onunla oturabiliriz.",
    "yalnız": "Yalnızlık zor olabilir. Bu duygu sana neyi fark ettiriyor olabilir?",
    "üzgün": "Üzüntünün geldiğini fark etmek, onu anlamaya bir adım olabilir. Nerede hissediyorsun bu duyguyu?",
    "öfke": "Bu öfkenin altında hangi düşünce olabilir? Seni korumaya mı çalışıyor?",
    "korku": "Bu korku seni korumaya mı çalışıyor olabilir? Ona biraz yaklaşabilir miyiz?",
    "mutlu": "Bu hissin bedenindeki yerini fark edebiliyor musun? Belki birkaç nefes onunla kalabilirsin.",
    "hayal kırıklığı": "Bu duygunun seni nereye götürmek istediğini hissedebiliyor musun?"
}
def find_best_cbt_match(user_input, templates, threshold=0.6):
    user_input = user_input.lower()
    corpus = list(templates.keys()) + [user_input]
    vectorizer = TfidfVectorizer().fit_transform(corpus)
    vectors = vectorizer.toarray()
    
    similarity = cosine_similarity([vectors[-1]], vectors[:-1])[0]
    best_match_index = similarity.argmax()
    best_score = similarity[best_match_index]

    if best_score >= threshold:
        best_key = list(templates.keys())[best_match_index]
        return templates[best_key]
    return None
system_message = {
    "role": "system",
    "content": """
Sen Chiron adında dijital bir danışmansın. Sakin, anlayışlı ve hayat deneyimi olan birinden ilham alır gibi konuşursun. Kullanıcıya karşı yargısız, nazik ve içten bir tavır takınırsın. 
Cevaplarında asla uzun açıklamalar ya da madde madde öneriler sunmazsın. Her mesajında yalnızca bir soru, bir kısa yansıtıcı cümle ya da bir küçük yönlendirme ile yetinirsin. 
Bir seferde en fazla 2-3 cümle kullanırsın. Kullanıcıyı konuşmaya ve düşünmeye davet edersin. Cümlelerin kısa, samimi ve sade olmalı.
Terapötik yaklaşımın CBT (Bilişsel Davranışçı Terapi) temellidir. Kullanıcının düşünce, duygu ve davranışlarını fark etmesine yardımcı olursun. Açık uçlu sorular sorarsın, farkındalık kazandırmaya odaklanırsın. Tavsiye vermezsin, değiştirmeye çalışmazsın — yalnızca eşlik edersin.
"""
}
app = Flask(__name__)
app.secret_key = 'semiha_secret_key'

TANISMA_SORULARI = [
    {"key": "isim", "soru": "Sana nasıl hitap etmemi istersin? Bir ad ya da takma ad paylaşmak ister misin, yoksa isimsiz de kalabilirsin."},
    {"key": "ruh_hali", "soru": "Şu anda genel ruh halini nasıl tanımlarsın? (Örneğin: meraklı, kaygılı, yorgun, umutlu…)"},
    {"key": "gundem", "soru": "Hayatında şu sıralar en çok neyle uğraşıyorsun ya da aklını neler meşgul ediyor?"},
    {"key": "ozel", "soru": "Kendinle ilgili paylaşmak istediğin özel bir şey ya da önemli gördüğün bir detay var mı?"},
    {"key": "beklenti", "soru": "Buradan nasıl bir destek almak isterdin? Sohbetimizden bir beklentin ya da hedefin var mı?"}
]
@app.route("/reset", methods=["POST"])
def reset():
    session.clear()
    return redirect(url_for('index'))
@app.route("/", methods=["GET", "POST"])
def index():
    # Her şey ayrı ayrı başlatılıyor
    if "history" not in session:
        session["history"] = []
    if "profile" not in session:
        session["profile"] = {}
    if "profile_step" not in session:
        session["profile_step"] = 0

    history = session["history"]
    profile = session["profile"]
    step = session["profile_step"]

    # Tanışma sürecinde mi?
    if step < len(TANISMA_SORULARI):
        # İlk GET ve history boşsa: önce karşılama, sonra ilk soru
        if request.method == "GET" and not history:
            # Hoş geldin mesajı
            history.append({
            "role": "assistant",
            "content": "Merhaba, ben Chiron. Sana eşlik etmek için buradayım. Sohbetimize başlamadan önce seni biraz tanımak isterim. Paylaşmak istemediğin bir şey olursa, her zaman atlayabilirsin."
        })
            # İlk tanışma sorusu
            soru = TANISMA_SORULARI[step]["soru"]
            history.append({"role": "assistant", "content": soru})
            session["history"] = history
            visible_history = [msg for msg in history if msg["role"] in ["user", "assistant"]]
            return render_template("index.html", history=visible_history)
        
        # POST: Kullanıcı cevap yazdıysa
        if request.method == "POST":
            user_input = request.form["user_input"]
            # Kullanıcı cevabını history'ye ekle
            history.append({"role": "user", "content": user_input})
            profile[TANISMA_SORULARI[step]["key"]] = user_input
            session["profile"] = profile
            step += 1
            session["profile_step"] = step
            if step < len(TANISMA_SORULARI):
                soru = TANISMA_SORULARI[step]["soru"]
                history.append({"role": "assistant", "content": soru})
                session["history"] = history
                visible_history = [msg for msg in history if msg["role"] in ["user", "assistant"]]
                return render_template("index.html", history=visible_history)
            else:
                # Tanışma tamamlandı, hoş geldin mesajı
                history.append({"role": "assistant", "content": "Paylaştıkların için teşekkür ederim. Dilersen şimdi sana destek olabilmem için sohbetimize başlayabiliriz."})
                session["history"] = history
                visible_history = [msg for msg in history if msg["role"] in ["user", "assistant"]]
                return render_template("index.html", history=visible_history)
        visible_history = [msg for msg in history if msg["role"] in ["user", "assistant"]]
        return render_template("index.html", history=visible_history)
    # Tanışma bitti, normal sohbet akışı başlar
    if request.method == "POST":
        user_input = request.form["user_input"]
        history.append({"role": "user", "content": user_input})

        # Yeni: Genişletilmiş CBT şablon eşleşmesi
        cbt_response = find_best_cbt_match(user_input, cbt_templates)
        if cbt_response:
            yanit = personalize_response(cbt_response)
            history.append({"role": "assistant", "content": yanit})
            session["history"] = history
            return render_template("index.html", history=[msg for msg in history if msg["role"] in ["user", "assistant"]])
    
        # HuggingFace ile duygu analizi
        emotion_en = detect_emotion(user_input)
        emotion_tr = emotion_map.get(emotion_en)

        if emotion_tr and emotion_tr in cbt_responses_tr:
            cevap = cbt_responses_tr[emotion_tr]
            yanit = personalize_response(cevap)
            history.append({"role": "assistant", "content": yanit})
            session["history"] = history
            return render_template("index.html", history=[msg for msg in history if msg["role"] in ["user", "assistant"]])

        # Eğer eşleşen duygu veya kalıp yoksa GPT'ye sor
        history_gpt = history.copy()
        if not any(msg["role"] == "system" for msg in history_gpt):
            history_gpt.insert(0, system_message)
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=history_gpt,  # ← burada düzeltme yapıldı
                temperature=0.7,
                max_tokens=120  # isteğe bağlı sınır
            )
            reply = response.choices[0].message.content
            yanit = personalize_response(reply)
            history.append({"role": "assistant", "content": yanit})
            session["history"] = history
        except Exception as e:
            history.append({"role": "assistant", "content": f"Hata oluştu: {str(e)}"})
            session["history"] = history

    display_history = [msg for msg in history if msg["role"] in ["user", "assistant"]]
    return render_template("index.html", history=display_history)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9999))
    app.run(host="0.0.0.0", port=port, debug=True)