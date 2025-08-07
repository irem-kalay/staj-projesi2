from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
from urllib.parse import urlparse, parse_qs
import time

#csvvvvvvvvvvvvvv
csv_file_path = 'video_links.csv'
try:
    df = pd.read_csv(csv_file_path)
    if 'url' not in df.columns or df.empty:
        print("CSV dosyasında 'url' sütunu yok veya dosya boş!")
        exit()
    urls = df['url'].dropna().tolist()
except Exception as e:
    print(f"CSV okunurken hata oluştu: {e}")
    exit()

# Video ID çıkarma fonksiyonu
def extract_video_id_from_url(url):
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

# YouTubeTranscriptApi nesnesi
ytt_api = YouTubeTranscriptApi()

#Süreyi ölçüyorum- başlangıç
start_time= time.time()

# Tüm URL’ler için döngü
for idx, youtube_url in enumerate(urls, 1):
    print(f"\n{'='*60}")
    print(f"[{idx}/{len(urls)}] İşleniyor: {youtube_url}")
    print(f"{'='*60}")

    video_id = extract_video_id_from_url(youtube_url)
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
                    start_time = entry.start
                    duration = entry.duration
                    text = entry.text
                    f.write(f"[{start_time:.2f} - {start_time + duration:.2f}] {text}\n")

            print(f"Transkript dosyaya yazıldı: {filename}")

    except Exception as e:
        print(f"Hata oluştu (Video ID: {video_id}): {e}")
        continue

end_time= time.time()
print("Süre: {(end_time - start_time).2f} saniye")