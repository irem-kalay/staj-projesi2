#import os
#from textsum.summarize import Summarizer
#türkçede çalışmıyor, bu güzel ama şuan
# === Ayarlar ===
#input_dir = "."  # Şu anki dizin
#summarizer = Summarizer()

# === Tüm .txt dosyalarını listele ===
#txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt") and "_summary" not in f and "_final_summary" not in f]

#for txt_file in txt_files:
#   print(f"📄 Özetleniyor: {txt_file}")

    # Özetleme işlemi
#   summary_path = summarizer.summarize_file(txt_file)

#   print(f"✅ Özet dosyaya yazıldı: {summary_path}\n")

# 1. NLTK ile Extractive Summarization (Cümle Skorlaması)
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import heapq


def nltk_summarize(text, num_sentences=3):
    # Gerekli NLTK verilerini indir (ilk çalıştırmada)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    # Cümlelere ayır
    sentences = sent_tokenize(text)

    # Kelime frekansı hesapla
    stop_words = set(stopwords.words('english'))
    word_frequencies = defaultdict(int)

    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for word in words:
            if word not in stop_words and word.isalpha():
                word_frequencies[word] += 1

    # Cümle skorları hesapla
    sentence_scores = defaultdict(float)
    for i, sentence in enumerate(sentences):
        words = word_tokenize(sentence.lower())
        for word in words:
            if word in word_frequencies:
                sentence_scores[i] += word_frequencies[word]

    # En yüksek skorlu cümleleri seç
    best_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    best_sentences.sort()

    return ' '.join([sentences[i] for i in best_sentences])


# 2. TextRank Algoritması ile Özetleme
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def textrank_summarize(text, num_sentences=3):
    sentences = sent_tokenize(text)

    if len(sentences) <= num_sentences:
        return text

    # TF-IDF vektörleri oluştur
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Benzerlik matrisi hesapla
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # PageRank algoritması uygula
    scores = np.ones(len(sentences))
    for _ in range(100):  # 100 iterasyon
        new_scores = np.ones(len(sentences))
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    new_scores[i] += similarity_matrix[i][j] * scores[j]
        scores = new_scores

    # En yüksek skorlu cümleleri seç
    ranked_sentences = sorted(((scores[i], i) for i in range(len(sentences))), reverse=True)
    selected_indices = sorted([ranked_sentences[i][1] for i in range(num_sentences)])

    return ' '.join([sentences[i] for i in selected_indices])


# 3. Transformers ile Modern AI Özetleme
from transformers import pipeline


def ai_summarize(text, max_length=150, min_length=50):
    # Hugging Face'den özetleme modeli yükle
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Uzun metinleri parçalara böl (BART max 1024 token)
    max_chunk = 1000
    chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]

    summaries = []
    for chunk in chunks:
        if len(chunk.strip()) > 100:  # Çok kısa parçaları atla
            summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            summaries.append(summary[0]['summary_text'])

    return ' '.join(summaries)


# 4. Basit Kelime Frekansı ile Özetleme (İyileştirilmiş)
def simple_frequency_summarize(text, num_sentences=3, min_sentence_length=10):
    sentences = sent_tokenize(text)

    # Çok kısa cümleleri filtrele
    sentences = [s for s in sentences if len(s.split()) > min_sentence_length]

    if len(sentences) <= num_sentences:
        return ' '.join(sentences)

    # Stop words listesi
    stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                      'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
                      'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might',
                      'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
                      'we', 'they', 'me', 'him', 'her', 'us', 'them', 'uh', 'um', 'so',
                      'like', 'just', 'going', 'want', 'get', 'now', 'here', 'there'])

    # Kelime frekanslarını hesapla
    word_freq = {}
    for sentence in sentences:
        words = sentence.lower().split()
        for word in words:
            # Sadece harflerden oluşan kelimeleri al ve stop words'leri çıkar
            clean_word = ''.join(c for c in word if c.isalpha())
            if clean_word and clean_word not in stop_words and len(clean_word) > 2:
                word_freq[clean_word] = word_freq.get(clean_word, 0) + 1

    # En sık kullanılan kelimeleri al (top %20)
    if not word_freq:
        return ' '.join(sentences[:num_sentences])

    max_freq = max(word_freq.values())
    important_words = {word: freq for word, freq in word_freq.items()
                       if freq >= max_freq * 0.3}  # En az %30 frekansta olanlar

    # Cümle skorlarını hesapla
    sentence_scores = {}
    for sentence in sentences:
        words = sentence.lower().split()
        score = 0
        word_count = 0

        for word in words:
            clean_word = ''.join(c for c in word if c.isalpha())
            if clean_word in important_words:
                score += important_words[clean_word]
                word_count += 1

        # Pozisyon bonusu (başlangıç cümleleri daha önemli)
        position_bonus = 1.0 - (sentences.index(sentence) / len(sentences)) * 0.3

        if word_count > 0:
            sentence_scores[sentence] = (score / word_count) * position_bonus

    if not sentence_scores:
        return ' '.join(sentences[:num_sentences])

    # En yüksek skorlu cümleleri seç
    best_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)

    # Orijinal sırayı koru
    result_sentences = []
    for sentence in sentences:
        if sentence in best_sentences:
            result_sentences.append(sentence.strip())

    return ' '.join(result_sentences)


# 5. YouTube/Podcast Transkripti için Özel Özetleme
def transcript_summarize(text, num_sentences=5):
    """YouTube videoları ve podcast transkriptleri için özel özetleme"""

    # Konuşma kalıplarını temizle
    text = text.replace('um ', '').replace('uh ', '').replace('like ', '')
    text = text.replace('[Music]', '').replace('[Applause]', '')

    sentences = sent_tokenize(text)

    # Teknik terimler ve anahtar kelimeler (video konusuyla ilgili)
    tech_keywords = ['python', 'pdf', 'summarize', 'langchain', 'gpt', 'claude', 'api',
                     'model', 'transformer', 'chain', 'document', 'load', 'install',
                     'code', 'function', 'file', 'key', 'anthropic', 'openai']

    # Cümle skorlama
    sentence_scores = {}
    for sentence in sentences:
        if len(sentence.split()) < 8:  # Çok kısa cümleleri atla
            continue

        words = sentence.lower().split()
        tech_score = sum(1 for word in words if any(keyword in word for keyword in tech_keywords))

        # Uzunluk skoru (çok kısa veya çok uzun cümlelere ceza)
        length_score = min(len(words) / 15, 1.0) if len(words) > 5 else 0.1

        # Pozisyon skoru (başlangıç ve sonun önemli olduğunu varsay)
        position = sentences.index(sentence) / len(sentences)
        position_score = 1.0 if position < 0.3 or position > 0.7 else 0.7

        sentence_scores[sentence] = tech_score * length_score * position_score

    if not sentence_scores:
        return ' '.join(sentences[:num_sentences])

    # En iyi cümleleri seç
    best_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)

    # Sıralama koru
    result = []
    for sentence in sentences:
        if sentence in best_sentences:
            result.append(sentence.strip())

    return ' '.join(result)


# KULLANIM ÖRNEKLERİ
if __name__ == "__main__":
    # Örnek metin
    sample_text = """
    İyi sandviç yapmayı bilirsen şu hayatta her şeyin üstesinden gelebilirsin. Neden? Çünkü basit malzemelerle bile iyi bir yemek ortaya çıkarıyorsun. Gözün doyuyor, yetiyor. Oysa hayat bazen elindeki en iyi malzemelerle bile sana kötü bir yemekmiş gibi görünebilir. Tam şurada bana aynen böyle görünüyordu. E bendeki malzeme bu. Bununla bir şeyler pişirmeye çalışıyorum işte sanki böyle farkında değilmişim gibi. Artist artist bozlar. Ama aslında kafamda arka plana girip çıkan o tipler var. Ne zaman boşaltacaklar, meydanı ne zaman bana bırakacaklar filan diye düşünüyorum. Mesela şu adam tam en iyi pozumu yakalamışım. Fotobomp yapıp duruyor. Kaç kere girdi çıktı bu kadraja ya. En sonunda tamam dedim ya pes ettim. Vazgeçtim bu fotoğraftan. Kötü bir yemek bu. Ama sonra ertesi gün çektiklerimi gözden geçirirken o adama bir daha baktım. Biraz daha yakından. Bu fotoğraftan 30 sene önce ona da hayatın kötü bir yemek gibi göründüğünü nereden bilecektim ki? Mesela en sevdiğim dizinin bir bölümünün bir parçasında palyaço kılığına bile girmek zorunda kaldığını bilemezdim. E ondaki malzeme de buymuş. Onunla yapabileceğinin en iyisini yapmayı denemiş. Aktör olmak istiyormuş. Yüzünün bile gözükmediği böyle rolleri kabul etmek zorunda kalmış. E sonra bakmış olmuyor. Kendi kaderinin kontrolünü kendi ellerine almış. Bir yemeğin içindeki malzeme olmaktansa o yemeği hazırlayan şef olmalıyım diye düşünmüş ve bunu yapmış. İşin mutfağına da geçip kendi projelerini yazmaya, kendi girişimlerini yapmaya, yönetmeye başlamış. Ve işte bu şekilde palyaçoluktan kurtulup Iron Man filmiyile başlayan o koskoca Marvel sinematik evreninin kurucularından biri haline gelmiş. Evet, hala bana fotobomp yapan aynı adamdan bahsediyorum. Eğer dikkatli bakarsanız onun Jean Favro olduğunu görebilirsiniz. Dünyanın en büyük Star Wars buluşmasında kapalı bir etkinlik sırasında çekilen bu fotoğrafta beni sinir eden adamın o olduğunu keşke daha önce anlayabilseydim. Çünkü oraya yine kendi projesi olan Mandalorean'ın yeni sezon tanıtımı için gelmişti. O artık dizilerde bir figür olmaktan çıkıp sinematik evrenler kurgulayan Hollywood'un en önemli yönetmenlerinden biri olmuştu. Yoksa şef mi demeliyiz? Evet, şef daha iyi bir tabir olur. Dünyanın en büyük restoranlarında en iyi malzemelerle görkemli yemekler hazırlayan bir şef gibi yönetmen. Ama hayat öyle bir şey ki iniş çıkışlarla dolu. Öyle sadece hamdım piştim olmuyor. Bazen yanmak da gerekiyor ki küllerinden tekrar doğabilesin. İşte Jean Favro'nun hayatında da aynen böyle bir dönem var. Onun grafiği Palyaço Erik karakteriyle ta en aşağıdan başlıyor ve Iron Man filmleriyle en tepeye kadar çıkıyor. Üstelik o filmlerde hem yönetmen hem de Iron Man'in şoförü Happy Hogan rolünde. Evet, karakterin adı Happy. Mutlu anlamına geliyor ama onu canlandıran Jean Favro hiç de mutlu değil o yıllarda. Demek ki zirvede bile kaybolabilmek mümkün. Devasa bütçeli filmlerdeki beklentiler hiç bitmiyor. Yüzlerce kişilik ekipler onu bunaltıyor. Her şeyi bir kenara bırakıp yeniden başlamak istiyor. Kendi hikayesini kendi kurallarıyla anlatma ihtiyacı bu. Onun grafiğindeki bu dönemeci çok iyi okumamız lazım. Çünkü hepimizi ilgilendiren önemli dersler var. Yani palyaço ya da aktör olmayı düşünmüyorsanız bile şimdi biraz daha kulak kabartın. Çünkü bazen gerçekten yanmak ve en başa, en temele dönmek gerekiyor ki küllerinizden asıl siz olarak doğabilesiniz. Jean Favro 2014'te süper kahramanlarla uğraşmaya bir ara verip kendi imkanlarıyla mütevazi bir film çekti. Şef bu filmi bana kısaca tanıt dersen eğer iyi sandviç yapmakla ilgili bir film derim. Çok basit bir hikaye. Öyle büyük çatışmalar yok. Aksiyon neredeyse sıfır. Romantizm yani biraz var ama arka planda. İşte o yüzden de yani tüm bu unsurların çok geri planda kalması nedeniyle çok da bilinmiyor bu film. Ama izleyince of sizi öyle bir yakalıyor ki tadı damağınızda kalıyor. İyi bir restoranda ünlü bir şef olarak çalışan birinin hikayesi bu. Üstelik teknolojiyle de bir ilgisi var. O yıllarda en önemli teknoloji trendi sosyal medyaydı. X'in Twitter olduğu yıllar hatta Twitter'ın Wine'i çıkardığı yıllar yani 6 saniyede meşhur olabildiğiniz zamanlar. İşte o zaman da şimdiki gibi teknoloji bazılarına fırsat sunuyor. Bazılarının da işini yok ediyordu. Klasik bıçak örneği. Birisini öldürebilir ve başka birini tedavi edebilir ya da bir şefin elinde nefis bir yemek hazırlayabilir. İşte o yılların yükselen teknolojisi. Sosyal medya bizim bu geleneksel şefin işlerini bozuyor. bir gurmenin yazdığı eleştiri yazısına Twitter DM'den özel bir cevap verdiğini sanıp o mesajı herkese açık yollayınca bir güzel linç yiyip yerine oturuyor ve bir günde hem işini hem de itibarını kaybediyor. >> E bakın 10 yıl sonra bugün de başka bir teknolojik tehditle karşı karşıyayız değil mi? Yapay zeka tehdidi. Mesleğimiz elimizden alınacak mı korkusu yaşıyoruz. Yapay zeka bizim yeni bıçağımız ama onu nasıl kullanmalıyız? Bizim her şeyini kaybeden o şefimiz sıfırdan başlamaya karar veriyor. Eskiden mutfakta bir sürü kişiyi yönetip sofistike menüler hazırlarken işinden olmuştu. Aynı şeyi denemek yerine başka bir restoranda aynı yöntemlerle yine karmaşık iş akışları kullanmak yerine kendisine yeni bir çıkış yolu ararken en temele inmeye karar veriyor. >> Eski eşinin yardımıyla eski bir yemek kamyonu buluyor. eski işi sırasında çokça ihmal ettiği küçük oğluyla o kamyonu temizleyip bir güzel adam ediyor ve eski bir arkadaşıyla birlikte sandviç yapmaya başlıyorlar. Eski eski deyip duruyorum çünkü bazı şeyler değişmiyor. Böylesine hızla değişen zamanlarda değişmeyen o eskileri bulup onlara tutunup onları yenilerle buluşturup bir füzyon yapmak gerekiyor. Aynı ikilem belki şu anda sizin önünüzde de var. Belki siz de tutkuyla bir şeyler yapmaya çalışıyorsunuz ama hep teknoloji engeliyle karşılaşıyorsunuz. Aslında biraz araştırırsanız o engeli aşabilecek, çözebilecek bir platform vardır belki. Yani hünerlerinizi sergileyebileceğiniz bir yemek kamyonunu bulabilirsiniz. İşte bu videonun sponsoru build store. Tam da bu food truck metaforunun e-ticaret dünyasındaki karşılığı gibi. Büyük bir e-ticaret sitesini kurabilmek için çevik bir şekilde yola çıkmanızı sağlıyor. Kendi ürünlerinizi satmaya hemen başlayabileceğiniz bir araç. Yapay zeka destekli algoritmalarla güçlendirilmiş bu aracı kullanarak e-ticaret dünyasına herkes girebilir. Sadece 2 dakika içinde seçtiğiniz kategoriye özel en çok satan 10 ürünle birlikte hazır bir online mağazaya sahip olabilirsiniz. Hem de çok kolay bir şekilde önce Build Your Store'a üye oluyor ilginizi çeken bir kategori seçiyorsunuz. Shopify üzerinden aylık 1 dolarlık başlangıç planına geçiyorsunuz. Ardından build your store uygulamasını kurup mağazanızı yapay zeka yardımıyla bir tıkla oluşturuyorsunuz. Oto DS'i entegre edip ürün ekleme ve sipariş yönetimini de otomatikleştirebiliyorsunuz. Ve işte mağazanız hazır bile. Az önceki şef nasıl kullanman gerektiğini öğreteceğim demişti ya. Build Your Store platformu da size satışa hazır ürün sayfaları ve ücretsiz dropshipping eğitimleri sunuyor. Normalde 200-250 dolar değerindeki premium temada ücretsiz olarak sizin oluyor. 40.000den fazla girişimcinin kullanmaya başladığı build your store üzerinden hemen şimdi bir Shopify mağazası açarsanız ilk 3 ay sadece aylık 1 dolar ödeyeceksiniz. Bu fırsatı kaçırmamak ve hızla e-ticaret sitenizi oluşturmak için açıklamalar bölümünde bu kanala özel bağlantıyı kullanmayı unutmayın. İşte hayattaki en büyük tutkusu yemek yapmak olan şefimiz kendisini böyle ayağa kaldırıyor. >> O yemek kamyonuyla bir yolculuğa çıkıyorlar. Hani hikayecilik de vardır ya kahramanın yolculuğu. Böyle başlıyor. Kent kent, sokak sokak dolaşıyorlar. Eskiye ait kendi yeteneklerini oğlunun temsil ettiği yeni nesil becerilerle buluşturuyor. Bir yandan sandviç yapıp satarken bir yandan sosyal medyayile bunun pazarlamasını yapıyorlar. hani o eski işini yok eden sosyal medyadan korkmak yerine Twitter'ı kullanarak kendine bir kitle oluşturuyor. Yani kendisini öldüren bu bıçağı bu kez yine kendisini diriltmek için kullanıyor. İşte en önemli mesele bu. Bıçağı iyi tanıyacaksın. Onu adeta bir uzantını haline getireceksin. Evet, gerçekten de öyle. Bir yandan bu bıçak çok keskin. Bazıları için tehlikeli. Teknoloji tam olarak böyle bir şey. İnternet çıktı, geleneksel ekonomiyi keskin bir bıçak gibi değiştirdi. Sosyal medya çıktı. Dikkatli olmayanlar orada kendini kaybetti. Ve şimdi yapay zeka var. O da bir bıçak. Onda da ustalaşmak gerekiyor. Ama nasıl? Yani bu senin işin. Geliştirmen gereken senin kendi yeteneğin. Altın bilezik gibi. Bunu bileğini bir kere taktın hangi mutfağa girersen gir artık hünerlerini göstermeye başlayabilirsin. O bıçağı keskin ve temiz tutup kaybetmemek artık senin sorumluluğun. Şimdi burada oğluyla bir bağ kurmaya çalışan bir baba figürü var gibi duruyor ama aslında her ikisi de birbirine bir şey öğretiyor. Baba oğluna yemek yapma becerisini eskiyi, oğlu da babasına onu sunma becerisini yeniyi veriyor. Sosyal medyayla nasıl paylaşabileceğini gösteriyor. Aradaki bıçak yani teknoloji sadece bu bağın kurulabilmesini sağlayan bir köprüden ibaret. Bugün olsa yapay zekayla onu daha iyi hale getirmenin yollarını arayıp bulurlardı. İşin aslı ne biliyor musunuz arkadaşlar? İşin aslı önce yemeğin içindekilere bakmak, malzemelere. Elde hangi malzeme var? Onları bir gözden geçir. Sonra ne pişireceğine karar ver. Tam bir girişimcilik dersi. Bazen en güzel yemekler en sade malzemelerle yapılır. Sadece doğru bileşimi bulmak gerekiyor. E hayat da böyle. Dedim ya en başta iyi sandviç yapmayı bilirsen şu hayatta her şeyin üstesinden gelebilirsin diye. Şimdi bu fotoğrafıma yeniden bakınca aslında o kadar da kötü bir yemek gibi görünmediğini düşünüyorum. Yani içindeki kılçıktan bile bir ders çıkarttım değil mi? Evet. Belki siz de fotoğraflarda benim gibi böyle boş boş bakıyor olabilirsiniz. Belki siz de kendinizi kaybolmuş hissediyor olabilirsiniz. Bütün bu teknolojik yenilikler, özellikle de yapay zeka sanki yerimizi alacakmış gibi geliyor olabilir. Yani yıllarca okudum, emek verdim, didindim ama şimdi işe yaramaz hale mi geleceğim? Hayır. O fotoğraftaki Jean Favro nasıl ki palyaçoluktan yönetmenliğe geçerken yeteneklerini hep biliyse bizim de bu çağda yapmamız gereken şey bu. Bıçağımızı bilemek. Ben bu kavramı Stephen Covy'den öğrenmiştim. Ormanda ağaç keserken durup baltayı bilemek için zaman ayırmak gerekiyor." diye yazmıştı Etkili İnsanların Yediği Alışkanlığı adlı kitabında. Görünürde bu bir zaman kaybı gibi gelebilir ama aslında uzun vadede en büyük kazancın yoludur. Bıçağı bilemek, yeteneklerinizi geliştirmek, teknolojik yetkinliğinizi arttırmak. Sormamız gereken soru bıçağımız ne? Belki insanlarla iletişim kurma yeteneğiniz, belki yaratıcı fikirler üretme gücünüz, belki sabrınız, belki öğrenme tutkunuz. O bıçağı nasıl daha keskin hale getirebilirsin? Hangi eğitimle, hangi deneyimle, hangi pratikle onu bileyle? Kim bilir belki senin iyi sandviçin bir podcast fikridir ya da bir dijital ürün, bir e-ticaret sitesidir. Şef filmindeki o food truck metaforunu unutmamak lazım. Çünkü o sıfırdan başlamak zorunda olan kahramanımıza küçük, çevik, hemen harekete geçebileceği bir başlangıç noktası görevini gördü. Yani aslında mesele şu. Bıçağı sen tutuyorsun. Elindeki malzemelere bakıyorsun ve şimdi yapman gereken tek şey de kendi food truck yolculuğuna çıkmak. İşin sırrı burada başlamakta. Hayatın hangi dönemecinde olursan ol yeniden başlamak mümkün. Filmdeki o eski külüstür kamyon gibi başlamak için mükemmel olmak zorunda da değilsin. Ama mükemmelleşmek için başlaman gerekiyor. Önemli olan yola çıkmak, yolda öğrenmek. Yapay zekada, sosyal medyada, hayatın getirdiği tüm zorluklarda sadece birer engel, birer bıçak, onları nasıl tutacağınız, onları neye dönüştüreceğiniz hep sizin elinizde. O yüzden şimdi elinizdeki malzemelere bir bakın. Bıçağınızı tekrar elinize alın. Onu bir güzel bileyin ve yapabileceğiniz en iyi sandviçi yapmaya başlayın. [Müzik]"""

    print("Orijinal Metin Uzunluğu:", len(sample_text.split()))
    print("\n" + "=" * 50)

    # 1. NLTK ile özetleme
    print("\n1. NLTK ile Özetleme:")
    summary1 = nltk_summarize(sample_text, 2)
    print(summary1)
    print(f"Özet Uzunluğu: {len(summary1.split())} kelime")

    # 2. TextRank ile özetleme
    print("\n2. TextRank ile Özetleme:")
    summary2 = textrank_summarize(sample_text, 2)
    print(summary2)
    print(f"Özet Uzunluğu: {len(summary2.split())} kelime")

    # 3. AI ile özetleme (yorum satırını kaldır ve transformers kur)
    # print("\n3. AI (BART) ile Özetleme:")
    # summary3 = ai_summarize(sample_text)
    # print(summary3)
    # print(f"Özet Uzunluğu: {len(summary3.split())} kelime")

    # 4. Basit frekans ile özetleme (iyileştirilmiş)
    print("\n4. İyileştirilmiş Frekans ile Özetleme:")
    summary4 = simple_frequency_summarize(sample_text, 2)
    print(summary4)
    print(f"Özet Uzunluğu: {len(summary4.split())} kelime")

    # 5. Transkript özetleme testi
    print("\n5. YouTube/Podcast Transkripti Özetleme:")
    # Bu örnek için gerçek transkript yerine sample_text kullanıyoruz
    summary5 = transcript_summarize(sample_text, 3)
    print(summary5)
    print(f"Özet Uzunluğu: {len(summary5.split())} kelime")

# GEREKLİ KÜTÜPHANELERİ KURMA
"""
Terminal'de şu komutları çalıştır:

pip install nltk
pip install scikit-learn
pip install numpy

# AI özetleme için (opsiyonel - büyük model):
pip install transformers torch

# Türkçe metin için:
pip install turkish-nlp
"""