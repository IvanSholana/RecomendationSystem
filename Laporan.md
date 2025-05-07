# **Laporan Proyek Machine Learning - Ivan Sholana**

## **Project Overview**

![image](https://github.com/user-attachments/assets/0148fe7e-73eb-4ffa-89c3-03ea682bd482)

Seiring dengan pesatnya pertumbuhan industri hiburan digital, khususnya platform streaming film, kebutuhan akan sistem rekomendasi yang efektif menjadi semakin penting. Jumlah film yang tersedia secara daring terus meningkat, sehingga pengguna seringkali mengalami kesulitan dalam menemukan film yang sesuai dengan preferensi mereka. Oleh karena itu, sistem rekomendasi hadir sebagai solusi untuk membantu pengguna menemukan konten yang relevan dan menarik secara lebih efisien. Salah satu pendekatan yang banyak digunakan dalam pengembangan sistem rekomendasi adalah content-based filtering.
Content-based filtering bekerja dengan menganalisis kesamaan fitur konten dari item yang direkomendasikan. Dalam konteks project ini yang menggunakan rekomendasi film, content-based filtering memanfaatkan atribut-atribut seperti
- judul
- genre
- sinopsis
- kata kunci (keywords)
- pemeran (cast)
- sutradara (director)

Untuk menambah kualitas rekomendasi yang diberikan digunakan metrik popularitas seperti rating dan jumlah voting. Dengan menganalisis fitur-fitur tersebut, sistem dapat mengidentifikasi kemiripan antara film yang telah disukai pengguna sebelumnya dengan film lainnya dalam basis data, sehingga dapat memberikan rekomendasi yang lebih personal. Keunggulan sistem rekomendasi berbasis konten adalah kemampuannya dalam menghasilkan rekomendasi yang relevan tanpa memerlukan data preferensi dari banyak pengguna lain. Sistem ini berfokus pada profil preferensi masing-masing pengguna berdasarkan riwayat interaksi atau film favoritnya. Dengan demikian, pendekatan ini sangat sesuai untuk mengatasi permasalahan cold start yang umum terjadi pada pengguna baru dalam sistem rekomendasi berbasis kolaboratif.

## **RISET TERKAIT**
Salah satu pendekatan yang banyak digunakan dalam pengembangan sistem rekomendasi adalah content-based filtering. 
Pendekatan ini bekerja dengan menganalisis kesamaan fitur konten dari item yang direkomendasikan (Adomavicius & Tuzhilin, 2005). 
Dalam konteks rekomendasi film, content-based filtering memanfaatkan atribut-atribut seperti judul, genre, sinopsis, kata kunci (keywords), pemeran (cast), dan sutradara (director). 
Penelitian oleh Pratama dan Pratomo (2023) menunjukkan bahwa penggunaan sinopsis film dengan algoritma TF-IDF dan cosine similarity dapat menghasilkan rekomendasi yang relevan berdasarkan kesamaan tekstual. 
Untuk meningkatkan kualitas rekomendasi, metrik popularitas seperti rating dan jumlah voting sering diintegrasikan, seperti yang ditunjukkan dalam sistem hibrida yang menggabungkan fitur konten dan popularitas untuk meningkatkan akurasi (Siregar et al., 2023).
Dengan menganalisis fitur-fitur tersebut, sistem dapat mengidentifikasi kemiripan antara film yang telah disukai pengguna sebelumnya dengan film lain dalam basis data, sehingga memberikan rekomendasi yang lebih personal. Keunggulan utama sistem rekomendasi berbasis konten adalah kemampuannya menghasilkan rekomendasi yang relevan tanpa memerlukan data preferensi dari banyak pengguna lain (Lops et al., 2011). 
Sistem ini berfokus pada profil preferensi masing-masing pengguna berdasarkan riwayat interaksi atau film favoritnya. Pendekatan ini sangat efektif untuk mengatasi permasalahan cold start yang umum terjadi pada pengguna baru dalam sistem rekomendasi berbasis kolaboratif, karena hanya memerlukan metadata film yang tersedia sejak awal (Schein et al., 2002). Sebagai contoh, penelitian oleh Rakesh (2023) menunjukkan bahwa sistem berbasis konten meningkatkan presisi dan recall sebesar 20% dan 25% untuk film baru atau niche yang memiliki sedikit data interaksi.

- Adomavicius, G., & Tuzhilin, A. (2005). Toward the next generation of recommender systems: A survey of the state-of-the-art and possible extensions. IEEE Transactions on Knowledge and Data Engineering, 17(6), 734-749. https://doi.org/10.1109/TKDE.2005.99
- Pratama, R. R., & Pratomo, A. W. (2023). Movie recommendation system using content-based filtering with TF-IDF and cosine similarity. International Journal on Information and Communication Technology, 9(1), 1-10. https://socjs.telkomuniversity.ac.id/ojs/index.php/ijoict/article/view/747
- Siregar, A. P., Nababan, A. A., & Gunawan, D. (2023). Hybrid movie recommender system based on content-based filtering and weighted average method. International Journal of Intelligent Systems and Applications, 15(4), 1-12. https://www.ijisae.org/index.php/IJISAE/article/view/3102
- Lops, P., de Gemmis, M., & Semeraro, G. (2011). Content-based recommender systems: State of the art and trends. In Recommender Systems Handbook (pp. 73-105). Springer. https://doi.org/10.1007/978-0-387-85820-3_3
- Schein, A. I., Popescul, A., Ungar, L. H., & Pennock, D. M. (2002). Methods and metrics for cold-start recommendations. In Proceedings of the 25th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 253-260). https://doi.org/10.1145/564376.564421
- Rakesh, S. (2023). Movie recommendation system using content-based filtering technique. Baghdad Journal for Engineering and Physical Sciences, 4(1), 1-10. https://bjeps.alkafeel.edu.iq/journal/vol4/iss1/7/

## **PENTINGNYA PROYEK**
Proyek ini penting untuk dikembangkan karena beberapa alasan di bawah ini:
1. **Meningkatkan Pengalaman Pengguna dalam Konsumsi Konten** :
  - Melalui sistem rekomendasi film, dapat membantu pengguna menemukan film yang sesuai dengan minat mereka tanpa harus mencari secara manual di antara ribuan judul.
  - Melalui pemanfaatan fitur seperti sinopsis, genre, pemeran, dan sutradara, sistem rekomendasi ini dapat memberikan rekomendasi yang lebih relevan sehingga meningkatkan kepuasan pengguna.
2. **Relevansi dalam Industri Hiburan Digital** :  
  - Saat ini, platform seperti Netflix, Amazon Prime, dan lainnya sangat bergantung pada sistem rekomendasi untuk mempertahankan pengguna. Proyek ini relevan karena menawarkan pendekatan berbasis konten yang dapat diintegrasikan ke dalam platform tersebut, terutama untuk pengguna baru yang belum memiliki riwayat tontonan yang mana pada kondisi ini **pendekatan berbasis kolaborasi kurang efektif.**

# **BUSINESS UNDERSTANDING**

## **Problem Statements**
Untuk mencapai pengembangan sistem rekomendasi film berbasis *content-based filtering* yang efektif dan berkualitas, proyek ini dirancang untuk menjawab beberapa permasalahan utama sebagai berikut:

1. Bagaimana merancang sistem rekomendasi yang dapat menghasilkan rekomendasi film yang relevan dan dipersonalisasi berdasarkan fitur-fitur konten seperti judul, sinopsis, genre, kata kunci, pemeran, dan sutradara?
2. Bagaimana menangani permasalahan kualitas data, seperti nilai yang hilang (misalnya genre kosong) dan inkonsistensi format, agar data yang digunakan dalam sistem rekomendasi tetap valid dan dapat diandalkan?
3. Bagaimana mengintegrasikan metrik popularitas seperti *weighted rating* dan *normalized popularity* ke dalam sistem rekomendasi untuk memastikan film yang direkomendasikan tidak hanya relevan tetapi juga berkualitas dan diminati banyak pengguna?

## **Goals**
Untuk mencapai tujuan di atas, proyek ini mengusulkan pendekatan content-based filtering dengan langkah-langkah berikut:

1. **Merancang Sistem Rekomendasi:**
    - Menggunakan dataset credits, keywords, dan movies_metadata untuk mengekstrak fitur seperti pemeran, sutradara, genre, sinopsis, dan kata kunci.
    - Mengatasi nilai genre yang hilang dengan klasifikasi NLP seperti pada goals nomor dua.
    - Mengintegrasikan dataset menggunakan merge berbasis kolom id untuk menciptakan dataset gabungan yang konsisten.
    - Menggunakan cosine similarity untuk menghitung kesamaan antara vektor TF-IDF (judul) dan BERT embeddings (fitur gabungan) dari film input dengan semua film dalam dataset.
    - Menggabungkan bobot fitur popularitas
    - Mengambil 5 film dengan skor terbaik.

2. **Penyelesaian Dataset Null:**
    - Menerapkan TF-IDF Vectorizer untuk judul film guna merepresentasikan feature kata-kata fitur sinopsis ke dalam tipe data angka.
    - Melakukan klasifikasi menggunakan multiclass classification
    - Melakukan fine-tunning model Pre-Trained Transofrmer
    - Melakukan perbandingan hasil antara metode TF-IDF dan hasil fine-tunning.
    - Mengisi nan value genre dengan metode terbaik.

3. **Integrasi Metriks Popularitas** :
    - Menghitung weighted rating menggunakan formula yang mempertimbangkan jumlah voting (vote_count) dan rata-rata rating (vote_average), serta nilai rata-rata global (C) dan ambang batas voting (m).
    - Menormalkan popularity untuk memastikan skala yang konsisten dalam perhitungan skor akhir.

## **Solution Statements**
  1. Menggunakan cosine similarity untuk menghitung kesamaan antara vektor TF-IDF (judul) dan BERT embeddings (fitur gabungan) dari film input dengan semua film dalam dataset.
  2. Mengembangkan fungsi rekomendasi yang menerima judul film sebagai input dan mengembalikan top-N film (misalnya, 5 film) dengan skor tertinggi berdasarkan kombinasi kesamaan konten, weighted rating, dan popularity.
  3. Menggabungkan skor kesamaan (TF-IDF dan BERT-based) dengan weighted rating dan normalized popularity menggunakan bobot tertentu (misalnya, 0.2 untuk masing-masing metrik) untuk menghasilkan skor akhir yang seimbang.

# **DATA UNDERSTANDING**
Data Source: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

**Credit Dataset**

![image](https://github.com/user-attachments/assets/5ebe2420-4534-4944-9ea4-862423877434)

| **Nama Fitur** | **Tipe Data** | **Deskripsi Singkat**                                                                 |
| -------------- | ------------- | ------------------------------------------------------------------------------------- |
| `cast`         | object        | Daftar pemeran (aktor/aktris) dalam film, biasanya disimpan dalam format JSON-like.   |
| `crew`         | object        | Daftar kru film, termasuk sutradara, penulis naskah, produser, dll. Format JSON-like. |
| `id`           | int / object  | ID unik film untuk menghubungkan data ini dengan dataset utama (`Movies Metadata`).   |


**DATA PREVIEW**

![image](https://github.com/user-attachments/assets/36a730e0-c365-44dc-af73-08ca67aa7c9b)

**Keyword Dataset**

![image](https://github.com/user-attachments/assets/f693469c-2c95-48af-90de-9dd291b5a937)

| **Nama Fitur** | **Tipe Data** | **Deskripsi Singkat**                                                                 |
| -------------- | ------------- | ------------------------------------------------------------------------------------- |
| `id`           | int64         | ID unik film yang digunakan untuk menghubungkan data ini dengan dataset utama.        |
| `keywords`     | object        | Kumpulan kata kunci deskriptif (tags) yang menggambarkan tema atau elemen utama film. |

**DATA PREVIEW**

![image](https://github.com/user-attachments/assets/ade0a675-0338-4483-adf6-71efcf677284)


**Movie Metada Dataset**

![image](https://github.com/user-attachments/assets/4feeaf41-ff85-42fb-a959-00de971b073c)
![image](https://github.com/user-attachments/assets/4768f60f-99d6-42c3-bd9e-e463c8e227cb)

| **Nama Fitur**          | **Tipe Data** | **Deskripsi Singkat**                                                              |
| ----------------------- | ------------- | ---------------------------------------------------------------------------------- |
| `adult`                 | object        | Menunjukkan apakah film ditujukan untuk penonton dewasa (`True` atau `False`).     |
| `belongs_to_collection` | object        | Informasi apakah film merupakan bagian dari koleksi/waralaba film tertentu.        |
| `budget`                | object        | Anggaran produksi film dalam USD (banyak nilai nol atau tidak diketahui).          |
| `genres`                | object        | Daftar genre film dalam format JSON-like (misal: Action, Comedy, Drama).           |
| `homepage`              | object        | URL resmi situs web film, jika tersedia.                                           |
| `id`                    | object        | ID unik film dalam database ini (kadang tidak konsisten formatnya).                |
| `imdb_id`               | object        | ID film pada basis data IMDb (contoh: tt1234567).                                  |
| `original_language`     | object        | Kode bahasa asli film (contoh: `en` untuk Inggris, `fr` untuk Prancis).            |
| `original_title`        | object        | Judul asli film seperti yang dirilis pertama kali (sebelum translasi/judul lokal). |
| `overview`              | object        | Ringkasan atau sinopsis film.                                                      |
| `popularity`            | object        | Skor popularitas film (bisa berasal dari interaksi pengguna, views, dll).          |
| `poster_path`           | object        | Path URL untuk poster film (digunakan dalam tampilan visual atau UI).              |
| `production_companies`  | object        | Daftar perusahaan produksi film dalam format JSON-like.                            |
| `production_countries`  | object        | Negara tempat produksi film dilakukan.                                             |
| `release_date`          | object        | Tanggal rilis film (formatnya bisa bervariasi, perlu normalisasi).                 |
| `revenue`               | float64       | Pendapatan total film dalam USD.                                                   |
| `runtime`               | float64       | Durasi film dalam menit.                                                           |
| `spoken_languages`      | object        | Daftar bahasa yang digunakan dalam film.                                           |
| `status`                | object        | Status film (misal: `Released`, `Post Production`, dll).                           |
| `tagline`               | object        | Kalimat promosi singkat yang biasanya digunakan sebagai slogan film.               |
| `title`                 | object        | Judul film versi akhir yang ditampilkan.                                           |
| `video`                 | object        | Menunjukkan apakah entri merupakan video film (boolean dalam string).              |
| `vote_average`          | float64       | Rata-rata nilai rating film dari pengguna.                                         |
| `vote_count`            | float64       | Jumlah total suara atau voting yang diberikan pengguna.                            |

**Data Explanatory**
**1. Genre Distribution**
Gambar di bawah adalah hasil explanatory genre pertama dalam kumpulan genre untuk tujuan memprediksi 1 genre untuk 1 film

![image](https://github.com/user-attachments/assets/a6a60cd5-f6a5-4fbd-afd4-8febd106d0f2)

- Hasil grafik di atas menunjukkan bahwa dataset mengalami unbalance yang tinggi sehingga akan sangat berpotensi mendapatkan hasil metriks evaluasi yang jelek untuk proses pelatihan model klasifikasi genre untuk imputasi nan value.

Gambar di bawah merupakan total distribusi genre yang ada dalam dataset

![image](https://github.com/user-attachments/assets/f6321272-ff67-4eae-b5c8-ec598f3dba92)

Hasil di atas menunjukkan bahwa jika ditotal dari setiap genre yang ada di dalam movie maka label tetap tidak seimbang sehingga akan mempengaruhi kualitas prediksi.

# Data Preparation

```python
import ast

def get_cast_names(cast_str):
    cast_list = ast.literal_eval(cast_str)
    names = [cast_member['name'] for cast_member in cast_list]
    return '; '.join(names)

def get_director(crew_str):
    crew_list = ast.literal_eval(crew_str)
    directors = [crew_member['name'] for crew_member in crew_list if crew_member['job'] == 'Director']
    return '; '.join(directors)

# Create new dataframe with cast names and director
credits_processed = pd.DataFrame({
    'id': credits['id'],
    'cast_names': credits['cast'].apply(get_cast_names),
    'director': credits['crew'].apply(get_director)
})
```

**PENJELASAN:**

**Fungsi `get_cast_names(cast_str)`**
Tujuan dari fungsi ini adalah untuk mengekstrak nama-nama pemeran dari data yang berisi informasi cast setiap film. Fungsi ini menyederhanakan informasi cast menjadi sebuah daftar nama pemeran yang dipisahkan oleh titik koma, memudahkan akses dan analisis data terkait aktor atau aktris yang terlibat dalam film.

**Fungsi `get_director(crew_str)`**
Tujuan dari fungsi ini adalah untuk mengekstrak nama sutradara dari data crew setiap film. Dengan fungsi ini, kita bisa memperoleh daftar nama sutradara yang terlibat dalam pembuatan film, dan menyajikannya dalam format yang lebih mudah diakses dan dianalisis.

**Membuat DataFrame Baru `credits_processed`**
Tujuan dari bagian ini adalah untuk membuat DataFrame baru yang menyertakan informasi penting dari kolom `cast` dan `crew`, yaitu nama-nama pemeran dan sutradara, dalam format yang lebih terstruktur. DataFrame baru ini mempermudah penggunaan informasi tersebut untuk analisis lebih lanjut atau untuk digunakan dalam sistem rekomendasi film.

```python
# Select important columns
important_columns = [
    'id', 'title', 'genres', 'overview',
    'vote_average', 'vote_count',
    'popularity', 'production_companies'
]
movies_metadata_selected = movies_metadata[important_columns].copy()
# Drop rows where overview is NaN
movies_metadata_selected = movies_metadata_selected.dropna(subset=['overview'])

movies_metadata_selected.head()
```
**PENJELASAN:**

Kode di atas bertujuan untuk mengambil fitur-fitur penting yang dinilai relevan dengan kebutuhan content-based filtering.

```python
def get_genre_names(genres_str):
    try:
        if pd.isna(genres_str['genres']):
            return genres_str['genres']  # Return NaN as is
        if genres_str['genres'] == '[]':
            return ''
        genres = ast.literal_eval(genres_str['genres'])
        return '; '.join([genre['name'] for genre in genres])
    except:
        return ''

def get_production_companies(production_companies_str):
    try:
        if pd.isna(production_companies_str['production_companies']):
            return production_companies_str['production_companies']  # Return NaN as is
        if production_companies_str['production_companies'] == '[]':
            return ''
        companies = ast.literal_eval(production_companies_str['production_companies'])
        return '; '.join([company['name'] for company in companies])
    except:
        return ''

movies_metadata_selected['genres'] = movies_metadata_selected.apply(get_genre_names, axis=1)
movies_metadata_selected['production_companies'] = movies_metadata_selected.apply(get_production_companies, axis=1)
```

**PENJELASAN:**

**Fungsi `get_genre_names`**
Tujuan dari fungsi ini adalah untuk mengekstrak dan menyajikan nama-nama genre film dalam format yang lebih sederhana dan mudah dibaca. Dengan mengubah data genre yang awalnya kompleks menjadi daftar nama genre yang dipisahkan titik koma, informasi genre setiap film menjadi lebih mudah diolah dan dianalisis.

**Fungsi `get_production_companies`**
Fungsi ini bertujuan untuk mengambil dan menyusun nama-nama perusahaan produksi yang terlibat dalam pembuatan film. Data yang semula tersimpan dalam bentuk tidak langsung dibaca, diubah menjadi daftar nama perusahaan yang tersusun rapi, sehingga mempermudah pemahaman dan penggunaan informasi tersebut dalam analisis film.

**Pengolahan Kolom pada `movies_metadata_selected`**
Bagian ini bertujuan untuk memperbarui data film dengan mengganti format genre dan perusahaan produksi menjadi lebih ringkas dan terstruktur. Hasilnya, informasi penting pada dataset menjadi lebih siap untuk digunakan dalam analisis data maupun sistem rekomendasi film.

```python
import ast

def extract_keywords(keyword_list):
    return ' '.join([kw['name'].replace(" ", "") for kw in keyword_list])

# Kalau datanya string JSON:
keywords['keywords'] = keywords['keywords'].apply(lambda x: extract_keywords(ast.literal_eval(x)))
```

**PENJELASAN:**

Kode di atas bertujuan untuk menggabungkan dataset keyword yang awalnya terpisah menjadi sebuah string panjang agar dapat disatukan menjadi prepresentasi dari fitur sebuah film.

## FILL NAN GENRE VALUES DENGAN TF-IDF

1. Mempersiapkan library dan data yang akan dilatih serta diprediksi
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Pisahkan data dengan genre yang ada dan yang kosong
df_with_genres = movies_metadata_selected[movies_metadata_selected['genres'].apply(len) > 0].copy()
df_without_genres = movies_metadata_selected[movies_metadata_selected['genres'].apply(len) == 0].copy()

```

2. Karena percobaan memprediksi banyak genre sekaligus mendapatkan hasil yang buruk maka dilakukan percobaan memprediksi satu genre saya. Oleh karena itu, di bawah terdapat `genre_one`.
```python
def parse_genres(genres_str):
    if pd.isna(genres_str) or genres_str == '':
        return []
    return [g.strip() for g in genres_str.split(';')]

df_with_genres['genres'] = df_with_genres['genres'].apply(parse_genres)

df_with_genres['genres_one'] = df_with_genres['genres'].apply(lambda x: x[0])

# Filter out rows where 'genres' contains any of the specified production companies
for company in ['Carousel Production', 'Aniplex', 'Odyssey Media']:
    df_with_genres = df_with_genres[~df_with_genres['genres_one'].str.contains(company, na=False)]
```

3. Melakukan preprocessing text yang dibutuhkan untuk kebutuhan proses pelatihan dan pembobotan pada proses embedding di tahap selanjutnya.
```python
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

df_with_genres['processed_overview'] = df_with_genres['overview'].apply(preprocess_text)
```

4. Melakukan embedding menggunakan metode TF-IDF dan melakukan splitting dataset menjadi test dan train.
```python
# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df_with_genres['processed_overview'])

# Label encoding
le = LabelEncoder()
y = le.fit_transform(df_with_genres['genres_one'])

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

5. Melakukan pelatihan model menggunakan multinomial karena ingin menghasilkan beberapa prediksi secara langsung.
```python
# Train model
model = MultinomialNB()
model.fit(X_train, y_train)
```

6. Melakukan evaluasi model menggunakan sejumlah matrix
```
# Predict on test set
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Precision, Recall, F1-score (macro = rata-rata antar kelas)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"F1 Score (macro): {f1:.4f}")

# Classification report (lebih detail per kelas)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
```

![image](https://github.com/user-attachments/assets/8e9464fe-43d8-4173-9532-ffd33eeeca97)

Hasil di atas menunjukkan bahwa model gagal memprediksi genre minoritas karena kualitas data yang inbalance. Hal tersebut dapat dilihat dari nilai recall dan F1-Scorenya.
Oleh karena itu dilakukan metode pendekatan kedua menggunakan cara yang lebih mutakhir.

## FILL NAN GENRE VALUES DENGAN TRANSFORMER
```python
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Daftar 'genre' yang sebenarnya adalah nama production company dan harus dihapus
fake_genres = [
    "Carousel Productions", "Vision View Entertainment", "Telescene Film Group Productions",
    "Aniplex", "GoHands", "BROSTA TV", "Mardock Scramble Production Committee",
    "Sentai Filmworks", "Odyssey Media", "Pulser Productions", "Rogue State", "The Cartel"
]

# Hapus baris jika ada salah satu item di fake_genres muncul di kolom genres
df_cleaned = df_with_genres[~df_with_genres['genres'].apply(lambda x: any(g in fake_genres for g in x))].reset_index(drop=True)

# Split string genres jadi list
movies_metadata_selected['genres_split'] = df_cleaned['genres']

# Explode agar setiap genre jadi baris sendiri
movies_metadata_selected_exploded = movies_metadata_selected.explode('genres_split').reset_index(drop=True)

# Daftar genre unik (sesuaikan dengan dataset Anda)
GENRES = movies_metadata_selected_exploded['genres_split'].dropna().unique()
NUM_LABELS = len(GENRES)

# Fungsi untuk mengubah list genre menjadi vektor biner
def genres_to_vector(genres_list, all_genres):
    vector = [1 if genre in genres_list else 0 for genre in all_genres]
    return vector

# Kelas TextDataset untuk Multilabel
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Fungsi Pelatihan
def train_model(model, data_loader, optimizer, device, pos_weight):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(data_loader)

# Fungsi Evaluasi
def eval_model(model, data_loader, device, thresholds=None):
    if thresholds is None:
        thresholds = [0.5] * NUM_LABELS
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = torch.zeros_like(probs)
            for i, threshold in enumerate(thresholds):
                preds[:, i] = (probs[:, i] > threshold).float()

            predictions.append(preds.cpu())
            true_labels.append(labels.cpu())

    predictions = torch.cat(predictions)
    true_labels = torch.cat(true_labels)
    macro_f1 = f1_score(true_labels.numpy(), predictions.numpy(), average='macro')
    micro_f1 = f1_score(true_labels.numpy(), predictions.numpy(), average='micro')
    accuracy_per_label = (predictions == true_labels).float().mean(dim=0)

    return total_loss / len(data_loader), macro_f1, micro_f1, accuracy_per_label

# Parameter
MAX_LEN = 256  # Sinopsis biasanya lebih panjang, jadi gunakan 256
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-5

df_with_genres['labels'] = df_with_genres['genres'].apply(lambda x: genres_to_vector(x, GENRES))
texts = df_with_genres['overview'].tolist()
labels = df_with_genres['labels'].tolist()

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Inisialisasi tokenizer dan model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification"
)

# Pindah model ke device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Inisialisasi dataset dan dataloader
train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = TextDataset(val_texts, val_labels, tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Hitung pos_weight untuk ketidakseimbangan
labels_np = np.array(labels)
pos_freq = np.mean(labels_np, axis=0)
neg_freq = 1 - pos_freq
pos_weight = torch.tensor(neg_freq / (pos_freq + 1e-10)).to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Loop pelatihan
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    train_loss = train_model(model, train_loader, optimizer, device, pos_weight)
    val_loss, macro_f1, micro_f1, acc_per_label = eval_model(model, val_loader, device)
    print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    print(f'Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}')
    print(f'Accuracy per genre: {dict(zip(GENRES, acc_per_label.numpy()))}')

# Simpan model
model.save_pretrained('fine_tuned_distilbert_genre')
tokenizer.save_pretrained('fine_tuned_distilbert_genre')
```

![image](https://github.com/user-attachments/assets/d4997630-0e75-4b5f-9a76-aa99062a4169)

Hasil di atas menunjukkan peningkatan macro F1 score yang sangat signifikan bahkan pada multilable class. Pada finetunning menghasilkan 59% sedangkan penggunaan TF-IDF dan MultinomialNB menghasilkan 14%.

**Predict Nan Value use Trained Model**
```python
# Function to tokenize text data
def tokenize_data(texts, tokenizer, max_length=128):
    encodings = tokenizer(
        texts.tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return encodings

# Tokenize the text data from df_without_genres
texts = df_without_genres['overview']  # Replace 'text' with your actual column name
encodings = tokenize_data(texts, tokenizer)

# Prepare inputs for the model
input_ids = encodings['input_ids'].to(device)
attention_mask = encodings['attention_mask'].to(device)

# Load the fine-tuned model and tokenizer
model_path = '/content/drive/MyDrive/fine_tuned_distilbert_genre'
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Function to tokenize text data
def tokenize_data(texts, tokenizer, max_length=128):
    encodings = tokenizer(
        texts.tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return encodings

# Tokenize the text data from df_without_genres
texts = df_without_genres['overview']  # Replace 'text' with your actual column name
encodings = tokenize_data(texts, tokenizer)

# Prepare inputs for the model
input_ids = encodings['input_ids'].to(device)
attention_mask = encodings['attention_mask'].to(device)

from torch.nn.functional import sigmoid

# Make predictions in batches (optional, for large datasets)
batch_size = 16
predicted_genres = []
with torch.no_grad():
    for i in range(0, len(input_ids), batch_size):
        batch_input_ids = input_ids[i:i + batch_size]
        batch_attention_mask = attention_mask[i:i + batch_size]
        outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
        logits = outputs.logits
        probs = sigmoid(logits)  # Sigmoid for multi-label probabilities
        batch_preds = (probs > 0.5).int().cpu().numpy()  # Threshold at 0.5
        for pred in batch_preds:
            genres = [GENRES[j] for j, val in enumerate(pred) if val == 1]
            predicted_genres.append(genres if genres else ['None'])

# Add predictions to the DataFrame
df_without_genres['predicted_genres'] = predicted_genres
```

**Hasil Prediksi**

![image](https://github.com/user-attachments/assets/6eb15734-edda-4f30-8531-45761b41cac1)

## Feature Engineering
```python
m = movies_metadata_selected['vote_count'].quantile(0.9)  # Minimum votes (top 10%)
C = movies_metadata_selected['vote_average'].mean()       # Mean vote across all movies
movies_metadata_selected['weighted_rating'] = movies_metadata_selected.apply(
    lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C),
    axis=1
)

movies_metadata_selected['vote_count_normalized'] = (movies_metadata_selected['vote_count'] - movies_metadata_selected['vote_count'].min()) / (movies_metadata_selected['vote_count'].max() - movies_metadata_selected['vote_count'].min())
```

**PENJELASAN:**

Tahapan ini berusaha memberikan peringkat lebih adil dengan mempertimbangkan baik kuantitas (jumlah vote) maupun kualitas (rating) sehingga menghindari bias terhadap film dengan sedikit vote namun rating tinggi, agar tidak mendominasi daftar film terbaik.

**Combine Dataset**
```python
combined_df = pd.merge(movies_metadata_selected, credits_processed, on='id', how='left')
combined_df = pd.merge(combined_df, keywords, on='id', how='left')

# Menggabungkan genre yang berbentuk list menjadi sebuah satu string dan menghapus movie yang tidak memiliki genre
combined_df['genres'] = combined_df['genres'].apply(lambda x: ' '.join(x))
combined_df = combined_df[combined_df['genres'].notna() & ~combined_df['genres'].isin(['', ' '])]

# Menggabungkan semua content yang dimiliki oleh feature untuk menjadi satu feature yang dapat mewakili semua feature secara bersamaan
combined_df['combined_features'] = (
    combined_df['overview'].fillna('') + ' ' +
    combined_df['keywords'] + ' ' +
    combined_df['genres'] + ' ' +
    combined_df['cast_names'].fillna('') + ' ' +
    combined_df['director'].fillna('')
)

content_based_recommendations = combined_df[['id', 'title', 'genres', 'combined_features','vote_count_normalized','popularity','weighted_rating']].copy()

# Menghapus data yang masih mengandung nan
content_based_recommendations = content_based_recommendations.dropna()
```

**Embedding Combined Feature**

Fitur yang telah dikombinasikan sebelumnya diembedding menggunakan 2 pendekatan yaitu transformer dan tf-idf yang akan disesuaikan bobot impactnya.
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Reset index to ensure proper alignment
content_based_recommendations = content_based_recommendations.reset_index(drop=True)

# TF-IDF for titles
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(content_based_recommendations['title'])

# Load model ke device
model_path = '/content/drive/MyDrive/fine_tuned_distilbert_genre'
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
bert_matrix = model.encode(content_based_recommendations['combined_features'].values, show_progress_bar=True)

# Convert bert_matrix to scipy sparse matrix for consistency
from scipy.sparse import csr_matrix

# Encode data (otomatis akan pakai GPU kalau device='cuda')
bert_matrix = model.encode(
    content_based_recommendations['combined_features'].values,
    show_progress_bar=True,
    device=device  # pastikan ini ditambahkan
)
```

# **MODELLING**

Fungsi recommend bertujuan untuk menghasilkan rekomendasi film berdasarkan kemiripan konten dengan film tertentu yang dijadikan acuan (berdasarkan judul). Untuk itu, fungsi ini menggabungkan dua pendekatan perhitungan kemiripan, yaitu menggunakan model berbasis TF-IDF dan model berbasis BERT, dengan pembobotan yang dapat disesuaikan melalui parameter alpha dan beta. Apabila judul film tidak ditemukan dalam data TF-IDF, maka sistem tetap memberikan rekomendasi dengan hanya mengandalkan representasi semantik dari BERT. Setelah skor kemiripan dihitung, fungsi ini menambahkan komponen evaluasi tambahan berupa popularitas dan rating film, yang telah dinormalisasi, guna memastikan film yang direkomendasikan tidak hanya relevan dari sisi konten tetapi juga memiliki kualitas dan popularitas tinggi. Akhirnya, fungsi ini mengurutkan hasil berdasarkan skor akhir dan mengembalikan sejumlah film teratas yang paling sesuai dengan preferensi yang dimaksud. Dengan pendekatan ini, sistem dapat memberikan hasil rekomendasi yang lebih akurat, personal, dan berkualitas tinggi.

```python
# Fungsi cari index dari judul input
def get_index_from_title(title):
    return content_based_recommendations[content_based_recommendations['title'].str.lower() == title.lower()].index[0]

# Fungsi similarity
def get_similarity_scores(index):
    tfidf_scores = cosine_similarity(tfidf_matrix[index], tfidf_matrix).flatten()
    bert_scores = cosine_similarity([bert_matrix[index]], bert_matrix).flatten()
    return tfidf_scores, bert_scores

def recommend(title, alpha=0.4, beta=0.6, top_n=5):
    try:
        index = get_index_from_title(title)
        tfidf_scores, bert_scores = get_similarity_scores(index)

        # Jika title ditemukan, gabungkan TF-IDF dan BERT
        content_score = alpha * tfidf_scores + beta * bert_scores

    except IndexError:
        # Kalau title tidak ditemukan, gunakan hanya BERT
        print(f"⚠️ Title '{title}' not found. Recommending based on content only.")
        # Kosongkan tfidf_scores agar alpha tidak digunakan
        tfidf_scores = np.zeros(len(content_based_recommendations))
        bert_scores = cosine_similarity([model.encode(title)], bert_matrix).flatten()
        content_score = bert_scores

    # Pastikan kolom numerik sudah bersih
    content_based_recommendations['popularity'] = pd.to_numeric(content_based_recommendations['popularity'], errors='coerce').fillna(0)
    normalized_rating = content_based_recommendations['weighted_rating'] / content_based_recommendations['weighted_rating'].max()
    normalized_popularity = content_based_recommendations['popularity'] / content_based_recommendations['popularity'].max()

    # Skor akhir
    final_score = content_score + 0.2 * normalized_rating + 0.2 * normalized_popularity

    # Urutkan dan ambil top-N
    top_indices = final_score.argsort()[-top_n-1:-1][::-1]


    return content_based_recommendations.iloc[top_indices][['title', 'genres','weighted_rating', 'popularity']]
```

# **EVALUATION**

Kode tersebut bertujuan untuk menganalisis hasil rekomendasi film berdasarkan genre yang muncul dari setiap judul film yang diberikan. Untuk setiap judul dalam daftar, sistem menghasilkan rekomendasi film dan mengambil informasi genre dari hasil tersebut. Genre yang awalnya berupa string kemudian diolah menjadi format list agar dapat dianalisis lebih lanjut. Dari kumpulan genre tersebut, sistem mengidentifikasi semua genre unik yang muncul, sehingga bisa diketahui keragaman genre dalam rekomendasi. Selanjutnya, sistem menghitung seberapa sering masing-masing genre muncul di antara film-film yang direkomendasikan, guna mengetahui genre mana yang paling dominan. Akhirnya, dari frekuensi kemunculan genre tersebut, sistem mencatat skor tertinggi sebagai representasi tingkat relevansi genre paling dominan dalam setiap hasil rekomendasi. Proses ini dapat membantu dalam evaluasi performa sistem rekomendasi serta memahami preferensi genre yang ditangkap oleh model.

Pengujian dilakukan menggunakan keywords berikut :
```python
title_keywords = [
    # Action/Adventure
    "mission", "quest", "journey", "escape", "chase", "battle", "war", "hero",
    "legend", "adventure", "survivor", "hunter", "rebel", "outlaw", "guardian",

    # Sci-Fi/Fantasy
    "star", "galaxy", "space", "time", "future", "alien", "robot", "machine",
    "dimension", "portal", "magic", "wizard", "dragon", "kingdom", "curse",

    # Drama/Romance
    "love", "heart", "dream", "life", "story", "soul", "forever", "kiss",
    "promise", "fate", "destiny", "summer", "autumn", "winter", "spring",

    # Comedy
    "funny", "crazy", "wild", "party", "road", "trip", "big", "bad",
    "super", "great", "misadventure", "buddy", "wedding", "night",

    # Horror/Thriller
    "dark", "night", "shadow", "fear", "ghost", "haunted", "evil", "dead",
    "scream", "blood", "curse", "mystery", "secret", "killer", "trap",

    # Crime/Mystery
    "murder", "crime", "detective", "case", "suspect", "thief", "gangster",
    "heist", "justice", "law", "order", "conspiracy", "truth", "lie",

    # Historical/Biography
    "king", "queen", "emperor", "warrior", "glory", "honor", "legacy",
    "rise", "fall", "empire", "revolution", "freedom", "battle", "hero",

    # Family/Animation
    "kid", "family", "friend", "dog", "cat", "bear", "lion", "prince",
    "princess", "adventure", "world", "magic", "toy", "dream",

    # Western
    "cowboy", "sheriff", "bandit", "desert", "gold", "frontier", "town",
    "duel", "rider", "trail", "sunset", "valley", "river",

    # Musical
    "song", "dance", "music", "band", "stage", "show", "star", "rhythm",
    "melody", "dream", "shine", "harmony", "sound",

    # General/Universal
    "last", "first", "new", "old", "lost", "found", "hidden", "broken",
    "forgotten", "end", "beginning", "home", "city", "island", "sky",
    "sea", "road", "path", "way", "man", "woman", "boy", "girl"
]
```

Lalu setiap keyword tersebut akan diuji menggunakan proses berikut:
```python
all_max_score = []

for title  in title_keywords:
  film_recomendation = recommend(title)

  # Ambil kolom genres
  retrieved_genres = film_recomendation['genres']

  # Pisahkan string genre menjadi list, tangani non-string
  all_genres = retrieved_genres.apply(lambda x: x.split(' ') if isinstance(x, str) else [])

  # Buat set dari semua genre unik, hindari string kosong
  all_unique_genres = set(genre for sublist in all_genres for genre in sublist if genre)

  # Inisialisasi dictionary untuk menyimpan frekuensi genre
  scores_dict = {genre: 0 for genre in all_unique_genres}

  # Hitung frekuensi setiap genre dengan cara lebih efisien
  for genres in all_genres:
      for genre in genres:
          if genre in scores_dict:
              scores_dict[genre] += 1

  # Ambil semua nilai frekuensi
  all_score = list(scores_dict.values())

  # Tentukan tingkatan relevansi berdasarkan frekuensi tertinggi
  max_score = max(all_score) if all_score else 0

  all_max_score.append(max_score)
```

Hasil akhir akan ditampilkan seperti berikut:
```python
total_score = 0

for score in all_max_score:
  total_score += score

average_score = total_score / len(all_max_score)
print(average_score)
```

Contoh hasil pengujian:

![image](https://github.com/user-attachments/assets/99648f15-1dc4-443d-99c4-d725f74c39de)




