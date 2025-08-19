from main import summarize_transcript, write_transcripts  # Flask dosyanın adı neyse onu yaz

results = summarize_transcript("video_links.csv")
for r in results:
    print(r)

print("\n=== Transkript Testi ===")
#results_transcript = write_transcripts("video_links.csv")
#for r in results_transcript:
#    print(r)


