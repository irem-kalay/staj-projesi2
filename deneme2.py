from youtube_transcript_api import YouTubeTranscriptApi #To be able to take transcripts from youtube
import pandas as pd #to parse youtube links i use pandas
from urllib.parse import urlparse, parse_qs
import time #to see how fast my code is using time
import requests
import re
from langdetect import detect #to detect the language while making summaries
import os

# ==== AYARLAR ====
csv_file_path = 'video_links.csv'
API_KEY = "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" #my free openrouter api key, (you can make it easily from the site: https://openrouter.ai/settings/keys)
#sınırlı, eğer sınır dolarsa yeni key alıp yapıştırıp çalıştırabilirsin yoksa ücretli

#taking url's from csv
try:
    df = pd.read_csv(csv_file_path)
    if 'url' not in df.columns or df.empty:
        print("there is no url row in csv or the file is empty")
        exit()
    urls = df['url'].dropna().tolist()
except Exception as e:
    print(f"an error happend while reading the file: {e}")
    exit()

#function to parse video ID of youtube video
def parse_video_url_for_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        if parsed_url.path == '/watch':
            query_params = parse_qs(parsed_url.query)
            return query_params.get('v', [None])[0]
        elif parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/embed/')[1]
    elif parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    return None

# Function summarizing with OpenRouter API
def summarize_with_openrouter(text):
    text_clean = re.sub(r"\[\d+\.\d+ - \d+\.\d+\]\s*", "", text) #transkript zamanlarını siliyor
    detected_lang = detect(text_clean)
    #print(f"Dil: {detected_lang}")

    if detected_lang == "tr":
        base_prompt = (
                "Aşağıdaki video sözlerini anlamlı, akıcı ve doğal bir paragraf halinde özetle. "
                "Sözlerin verdiği duyguyu da yansıt:\n\n" + text_clean
        )
    elif detected_lang == "en":
        base_prompt = (
                "Summarize the following lyrics in a meaningful, fluent, and natural paragraph. "
                "Try to also reflect the emotion of the video:\n\n" + text_clean
        )
    else:
        base_prompt = (
                f"Summarize the following text in {detected_lang} in a meaningful, fluent, and natural paragraph. "
                "Try to also reflect the emotion of the video:\n\n" + text_clean
        )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://openrouter.ai",
        "X-Title": "Lyrics Summarizer"
    }

    #Claude-3-haiku
    data_claude = {
        "model": "anthropic/claude-3-haiku",
        "messages": [{"role": "user", "content": base_prompt}]
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data_claude
        )
        response.raise_for_status()
        res_json = response.json()
        if "choices" in res_json:
            return res_json["choices"][0]["message"]["content"]
        else:
            print("Claude modelinden beklenmedik yanıt:", res_json)
    except Exception as e:
        print(f"Claude modelinde hata: {e}")

    #Mistral eğer claude hatalıysa
    humanized_prompt = (
            "Sen sıcak, doğal ve insan gibi konuşan bir asistan gibisin. "
            "Cevaplarında duyguları hissettir, kısa cümleler kullan ama akıcı ol. "
            "Resmi değil, samimi bir tonla anlat. "
            + base_prompt
    )

    data_mistral = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [{"role": "user", "content": humanized_prompt}]
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data_mistral
        )
        response.raise_for_status()
        res_json = response.json()
        if "choices" in res_json:
            return res_json["choices"][0]["message"]["content"]
        else:
            return f"Mistral modelinden beklenmedik yanıt: {res_json}"
    except Exception as e:
        return f"Mistral modelinde hata: {e}"


# YouTubeTranscriptApi object
ytt_api = YouTubeTranscriptApi()

#calculating the time, start_time
start_time = time.time()

#loop for all urls, 1den başlıyor çünkü 0. satırda "url" başlığı yazılı
for idx, youtube_url in enumerate(urls, 1):
    print(f"\n{'='*60}")
    print(f"[{idx}/{len(urls)}] İşleniyor: {youtube_url}")
    print(f"{'='*60}")

    video_id = parse_video_url_for_id(youtube_url)
    if not video_id:
        print(f"Video ID çıkarılamadı: {youtube_url}")
        continue

    print(f"Çıkarılan Video ID: {video_id}")

    try:
        transcript_list = ytt_api.list(video_id)

        for transcript in transcript_list:
            print(
                transcript.video_id,
                transcript.language,
                transcript.language_code,
                transcript.is_generated,
                transcript.is_translatable,
                transcript.translation_languages,
            )

            transcript_data = transcript.fetch()

            filename = f"{video_id}_{transcript.language_code}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                for entry in transcript_data:
                    start_t = entry.start
                    duration = entry.duration
                    text = entry.text
                    f.write(f"[{start_t:.2f} - {start_t + duration:.2f}] {text}\n")

            print(f"Transkript dosyaya yazıldı: {filename}")

            #read and summarize
            with open(filename, "r", encoding="utf-8") as f:
                transcript_text = f.read()

            summary = summarize_with_openrouter(transcript_text)

            #summary file is here
            summary_filename = os.path.splitext(filename)[0] + "_summary.txt"
            with open(summary_filename, "w", encoding="utf-8") as sf:
                sf.write(summary)

            print(f" Özet yazıldı: {summary_filename}")


    except Exception as e:
        print(f"Hata oluştu (Video ID: {video_id}): {e}")
        continue

end_time = time.time()
print(f"Süre: {end_time - start_time:.2f} saniye")
