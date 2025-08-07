from transformers import pipeline

# === AYARLAR ===
input_file = "rYEDA3JcQqw_en.txt"
output_file = "rYEDA3JcQqw_en_summary.txt"

# === MODELİ YÜKLE ===
summarizer = pipeline("summarization", model="google/pegasus-xsum")

# === DOSYAYI OKU ===
with open(input_file, "r", encoding="utf-8") as f:
    full_text = f.read().strip()

# === ÖZETLE ===
summary = summarizer(full_text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']

# === SONUCU GÖSTER ===
print("\n--- Özet ---\n")
print(summary)

# === DOSYAYA YAZ ===
with open(output_file, "w", encoding="utf-8") as f:
    f.write(summary)

print(f"\n✅ Özet '{output_file}' dosyasına yazıldı.")
