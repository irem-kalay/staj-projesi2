#Frontend için API yapıyorum
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
from urllib.parse import urlparse, parse_qs
import re, requests, time
from langdetect import detect
import requests
from youtube_transcript_api import YouTubeTranscriptApi

#Engellenen IP'de işe yaramıyor
#proxies = {
#    "http": "http://213.233.178.137:3128",
#    "https": "http://42.119.98.66:16000"
#}

#session = requests.Session()
#session.proxies.update(proxies)


# Proxy kontrolü
#try:
 #   res = session.get("https://api.ipify.org?format=json", timeout=5)
 #   ip = res.json().get("ip")
 #   print(f"Proxy ile bağlanıyor, görünen IP: {ip}")
#except Exception as e:
 #   print(f"Proxy çalışmıyor veya erişilemiyor: {e}")


# monkeypatch YouTubeTranscriptApi içindeki requests oturumunu
#_cli.requests = session
ytt_api = YouTubeTranscriptApi()

API_KEY = "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"  # kendi anahtarın

app = Flask(__name__)
CORS(app)

#ytt_api = YouTubeTranscriptApi()

# === Yardımcı Fonksiyonlar ===
def parse_video_url_for_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query).get('v', [None])[0]
        elif parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/embed/')[1]
    elif parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    return None

def summarize_text_with_ai(text, ai_model="claude"):
    """
    text: özetlenecek metin
    ai_model: frontend'den seçilen AI. Örn: "openrouter", "gpt", "claude", vb.
    """
    text_clean = re.sub(r"\[\d+\.\d+ - \d+\.\d+\]\s*", "", text)
    detected_lang = detect(text_clean)

    if detected_lang == "tr":
        base_prompt = "Aşağıdaki metni anlamlı bir paragraf olarak özetle:\n\n" + text_clean
    else:
        base_prompt = "Summarize the following text:\n\n" + text_clean

    if ai_model == "deepseek":
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek/deepseek-r1:free",
            "messages": [{"role": "user", "content": base_prompt}]
        }
        try:
            res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Hata: {e}"
    elif ai_model == "gpt": #frontend'de adı gpt
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "openai/gpt-oss-20b:free",  # OpenRouterdaki ücretsiz gpt modelinin adıydı
            "messages": [{"role": "user", "content": base_prompt}]
        }
        try:
            res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Hata: {e}"

    elif ai_model== "mistral":
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "mistralai/mistral-small-3.2-24b-instruct:free",  # OpenRouterdaki ücretsiz gpt modelinin adıydı
            "messages": [{"role": "user", "content": base_prompt}]
        }
        try:
            res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Hata: {e}"

    elif ai_model== "gemini":
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "google/gemini-2.0-flash-exp:free",  # OpenRouterdaki ücretsiz gemini modelinin adıydı
            "messages": [{"role": "user", "content": base_prompt}]
        }
        try:
            res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Hata: {e}"


# === Sadece Orijinal Transkript Çıkaran Fonksiyon ===
def write_transcripts(csv_path):
    results = []
    start_time = time.time()

    try:
        df = pd.read_csv(csv_path)
        if 'url' not in df.columns or df.empty:
            raise ValueError("CSV dosyasında 'url' sütunu yok veya dosya boş")
        urls = df['url'].dropna().tolist()
    except Exception as e:
        raise RuntimeError(f"CSV okunamadı: {e}")

    for idx, youtube_url in enumerate(urls, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(urls)}] İşleniyor: {youtube_url}")
        print(f"{'='*60}")

        video_id = parse_video_url_for_id(youtube_url)
        if not video_id:
            results.append({"url": youtube_url, "error": "Video ID çıkarılamadı"})
            continue

        print(f"Çıkarılan Video ID: {video_id}")

        try:
            transcript_list = ytt_api.list(video_id)

            # Önce orijinal transkripti bul
            original_transcript = None
            for transcript in transcript_list:
                if not transcript.is_translatable:  # genelde orijinal dil
                    original_transcript = transcript
                    break

            # Eğer yukarıda bulunmazsa fallback: otomatik oluşturulan (speech recognition) olanı seç
            if not original_transcript:
                for transcript in transcript_list:
                    if transcript.is_generated:
                        original_transcript = transcript
                        break

            if not original_transcript:
                raise ValueError("Orijinal dilde transkript bulunamadı")

            # Transkripti al
            transcript_data = original_transcript.fetch()
            transcript_text = "\n".join(
                f"[{entry.start:.2f} - {entry.start + entry.duration:.2f}] {entry.text}"
                for entry in transcript_data
            )

            print(f"Orijinal transkript alındı: {video_id} ({original_transcript.language_code})")

            results.append({
                "url": youtube_url,
                "video_id": video_id,
                "language": original_transcript.language_code,
                "transcript": transcript_text
            })

        except Exception as e:
            error_msg = f"Hata oluştu (Video ID: {video_id}): {e}"
            print(error_msg)
            results.append({"url": youtube_url, "error": str(e)})
            continue
        # ✅ Her istekten sonra 30 saniye bekle
        if idx < len(urls):
            print("20 saniye bekleniyor...")
            time.sleep(20)

    end_time = time.time()
    print(f"Süre: {end_time - start_time:.2f} saniye")
    return results

#write_transcripts fonksiyonuyla transkripti çekiyor, summarize_text_with_ai fonksiyonuyla özet çıkartıyor
def summarize_transcript(csv_path, ai_model="openrouter"):
    transcripts_results = write_transcripts(csv_path)
    results = []
    start_time = time.time()

    for item in transcripts_results:
        if "error" in item:
            results.append(item)
            continue

        transcript_text = item["transcript"]
        video_id = item["video_id"]
        language_code = item["language"]

        # Özetleme
        summary = summarize_text_with_ai(transcript_text, ai_model)

        print(f"Özet oluşturuldu: {video_id} ({language_code})")

        results.append({
            "url": item["url"],
            "video_id": video_id,
            "language": language_code,
            "summary": summary
        })

    end_time = time.time()
    print(f"Süre: {end_time - start_time:.2f} saniye")
    return results




# === API Endpoint ===
@app.route("/process", methods=["POST"])
def process_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "CSV dosyası yüklenmedi"}), 400

    file = request.files['file']
    save_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(save_path)

    # DOĞRU
    ai_model = request.form.get("aiModel", "claude")  # Anahtar adını "aiModel" olarak düzeltin.
    # .get() metodunun ikinci parametresiyle daha temiz bir varsayılan değer atayın.
    try:
        results = summarize_transcript(save_path, ai_model)  # artık seçilen AI kullanılıyor
        return jsonify({"status": "ok", "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# === Sadece Transkript Endpoint ===
@app.route("/transcripts", methods=["POST"])
def transcripts_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "CSV dosyası yüklenmedi"}), 400

    file = request.files['file']
    save_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(save_path)

    try:
        results = write_transcripts(save_path) #burda çağırıyorum
        return jsonify({"status": "ok", "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)


