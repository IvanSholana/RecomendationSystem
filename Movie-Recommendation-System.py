import pandas as pd

credits = pd.read_csv('/content/drive/MyDrive/Project/Dicoding Recomendation System/credits.csv')
keywords = pd.read_csv('/content/drive/MyDrive/Project/Dicoding Recomendation System/keywords.csv')
movies_metadata = pd.read_csv('/content/drive/MyDrive/Project/Dicoding Recomendation System/movies_metadata.csv')

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

# Convert id to numeric, coercing errors to NaN
movies_metadata['id'] = pd.to_numeric(movies_metadata['id'], errors='coerce')
# Drop rows where id is NaN
movies_metadata = movies_metadata.dropna(subset=['id'])
# Convert id to integer
movies_metadata['id'] = movies_metadata['id'].astype(int)

credits['id'] = pd.to_numeric(credits['id'], errors='coerce')

keywords['id'] = pd.to_numeric(keywords['id'], errors='coerce')

# Select important columns
important_columns = [
    'id', 'title', 'genres', 'overview',
    'vote_average', 'vote_count',
    'popularity', 'production_companies'
]
movies_metadata_selected = movies_metadata[important_columns].copy()

# Drop rows where overview is NaN
movies_metadata_selected = movies_metadata_selected.dropna(subset=['overview'])

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

import ast

def extract_keywords(keyword_list):
    return ' '.join([kw['name'].replace(" ", "") for kw in keyword_list])

# Kalau datanya string JSON:
keywords['keywords'] = keywords['keywords'].apply(lambda x: extract_keywords(ast.literal_eval(x)))

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

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Pisahkan data dengan genre yang ada dan yang kosong
df_with_genres = movies_metadata_selected[movies_metadata_selected['genres'].apply(len) > 0].copy()
df_without_genres = movies_metadata_selected[movies_metadata_selected['genres'].apply(len) == 0].copy()

def parse_genres(genres_str):
    if pd.isna(genres_str) or genres_str == '':
        return []
    return [g.strip() for g in genres_str.split(';')]

df_with_genres['genres'] = df_with_genres['genres'].apply(parse_genres)

df_with_genres['genres_one'] = df_with_genres['genres'].apply(lambda x: x[0])

# Filter out rows where 'genres' contains any of the specified production companies
for company in ['Carousel Production', 'Aniplex', 'Odyssey Media']:
    df_with_genres = df_with_genres[~df_with_genres['genres_one'].str.contains(company, na=False)]
    
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

df_with_genres['processed_overview'] = df_with_genres['overview'].apply(preprocess_text)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df_with_genres['processed_overview'])

# Label encoding
le = LabelEncoder()
y = le.fit_transform(df_with_genres['genres_one'])

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

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

# View the results
df_without_genres[['overview', 'predicted_genres']]
movies_metadata_selected = pd.concat([df_with_genres, df_without_genres], ignore_index=True)

m = movies_metadata_selected['vote_count'].quantile(0.9)  # Minimum votes (top 10%)
C = movies_metadata_selected['vote_average'].mean()       # Mean vote across all movies
movies_metadata_selected['weighted_rating'] = movies_metadata_selected.apply(
    lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C),
    axis=1
)

movies_metadata_selected['vote_count_normalized'] = (movies_metadata_selected['vote_count'] - movies_metadata_selected['vote_count'].min()) / (movies_metadata_selected['vote_count'].max() - movies_metadata_selected['vote_count'].min())

combined_df = pd.merge(movies_metadata_selected, credits_processed, on='id', how='left')
combined_df = pd.merge(combined_df, keywords, on='id', how='left')

combined_df['genres'] = combined_df['genres'].apply(lambda x: ' '.join(x))
combined_df = combined_df[combined_df['genres'].notna() & ~combined_df['genres'].isin(['', ' '])]

combined_df['combined_features'] = (
    combined_df['overview'].fillna('') + ' ' +
    combined_df['keywords'] + ' ' +
    combined_df['genres'] + ' ' +
    combined_df['cast_names'].fillna('') + ' ' +
    combined_df['director'].fillna('')
)


content_based_recommendations = combined_df[['id', 'title', 'genres', 'combined_features','vote_count_normalized','popularity','weighted_rating']].copy()
content_based_recommendations = content_based_recommendations.dropna()

from sentence_transformers import SentenceTransformer, util
import torch

# Cek apakah GPU tersedia
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

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
  
total_score = 0

for score in all_max_score:
  total_score += score

average_score = total_score / len(all_max_score)
average_score


