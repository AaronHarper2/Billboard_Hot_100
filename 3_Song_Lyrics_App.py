from flask import Flask, render_template_string, request
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from unidecode import unidecode
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import os

app = Flask(__name__)

csv_filename = 'Billboard100.csv'
csv_path = os.path.join(os.path.dirname(__file__), csv_filename)

Billboard = pd.read_csv(csv_path)

def scrape_lyrics(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "lxml")
        lyrics_element = soup.find("div", class_="ltf")
        if lyrics_element:
            lyrics = lyrics_element.get_text(separator="\n")
            return lyrics.strip() 
    return None

def generate_url(artist, song):
    artist = artist.replace(" ", "-").lower()
    song = song.replace(" ", "-").lower()
    return f"https://lyricstranslate.com/en/{artist}-{song}-lyrics.html"

def clean_song(x):
    x = x.replace("-", " ")
    x = x.replace("'", "")
    x = re.sub(r'[^\w\s]', '', x)
    x = unidecode(x)
    unwanted_words = ['the', 'like', 'a', 'with', 'in', 'for', 'up', 'to', 'at', 
                      'on', 'that', 'from', 'of', 'but', 'as', 'before', 'is', 'by']
    words = x.split()
    cleaned_words = [word for word in words if word.lower() not in unwanted_words]
    return ' '.join(cleaned_words).lower()

def clean_lyrics(lyrics):
    lyrics = re.sub(r'\[.*?\]', '', str(lyrics))
    lyrics = lyrics.replace('\n', ' ')
    return lyrics

def most_frequent_word(lyrics):
    words = re.findall(r'\b\w+\b', lyrics.lower())
    words = [word for word in words if word not in stop_words]
    return max(set(words), key=words.count)

def normalize_document(doc):
    if pd.isna(doc):
        return ''
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    tokens = nltk.word_tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)

stop_words = nltk.corpus.stopwords.words('english')

index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Search For A Song</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f3f3f3;
        }
        .container {
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
            color: #333;
        }
        h3 {
            text-align: center;
            color: #333;
            font-size: 14px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input[type="text"] {
            width: 100%;
            padding: 8px;
            margin-bottom: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        input[type="submit"] {
            width: 100%;
            padding: 10px;
            background-color: #ff8200;
            color: #fff;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        input[type="submit"]:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Search For A Song</h1>
        <h3>Please enter an artist and title below of any song. This app will retrieve and provide information regarding the song's lyrics. If the app does not return any information, check for misspellings otherwise the lyrics for the song may be unavailable. </h3>
        <br>
        <form action="/search" method="post">
            <label for="artist">Enter artist:</label>
            <input type="text" id="artist" name="artist" required><br><br>
            <label for="song">Enter song:</label>
            <input type="text" id="song" name="song" required><br><br>
            <input type="submit" value="Search">
        </form>
    </div>
</body>
</html>
"""

result_html = """
<!DOCTYPE html>
<html>
<head>
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
  <meta http-equiv="Content-Style-Type" content="text/css">
  <title>Search Results</title>
  <style type="text/css">
    body {
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 0;
      background-color: #f3f3f3;
    }
    h1, h3, p, ul, li {
      margin: 0;
      padding: 0;
      list-style: none;
      color: #333;
      font-size: 16px; 
    }
    h1 {
      text-align: center;
      font-size: 24px;
    }
    h2 {
      text-align: center;
      font-size: 28px; 
    }
    h3 {
      text-align: center;
    }
    p, ul {
      text-align: center;
    }
    li {
      font: 16px Arial;
    }
    .return-btn {
      display: block;
      margin: 20px auto;
      padding: 8px 16px;
      width: 20%;
      background-color: #ff8200;
      color: #fff;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      text-decoration: none;
      text-align: center;
    }
    .return-btn:hover {
      background-color: #45a049;
    }
  </style>
</head>
<body>
<br>
<h2>{{ song_input.title() }} by {{ artist_input.title() }}</h2>
<br>
<br>
<h3>Total Words In This Song:<h3>
  <li>{{ word_count_new_song }}<li>
<br>
<br>
<h3>Most Frequent Word In This Song:<h3>
  <li>{{ most_frequent_word_new_song }}<li>
<br>
<br>
<h3>Words Similar To '{{ most_frequent_word_new_song.title() }}' That Appear In Popular Songs:</h3>
<ul>
  {% for word, similarity in similar_words %}
    <li>{{ word }}</li>
  {% endfor %}
</ul>
<br>
<br>
<h3>Popular Songs With Similar Lyrics:</h3>
<ul>
  {% for song in similar_songs %}
  <li>{{ loop.index }}. {{ song }}</li>
  {% endfor %}
</ul>
<br>
<a href="/" class="return-btn">Return to Search</a>
</body>
</html>
"""

@app.route("/")
def index():
    return index_html

@app.route("/search", methods=['POST'])
def search():
    global Billboard
    artist_input = request.form['artist'].replace('the ', '')
    song_input = clean_song(request.form['song'])
    url = generate_url(artist_input, song_input)
    lyrics = scrape_lyrics(url)

    if lyrics:
        cleaned_lyrics = clean_lyrics(lyrics)
        new_row = pd.DataFrame({'Artist': [artist_input], 'Title': [song_input], 'Lyrics': [cleaned_lyrics]})
        Billboard = pd.concat([Billboard, new_row], ignore_index=True)
        word_count_new_song = len(cleaned_lyrics.split())
        most_frequent_word_new_song = most_frequent_word(cleaned_lyrics)

        lyrics = Billboard['Lyrics'].dropna().values.tolist()
        lyrics_tokenized = [lyric.split() for lyric in lyrics]

        model = Word2Vec(sentences=lyrics_tokenized, vector_size=100, window=5, min_count=5, workers=4)

        focal_words = [most_frequent_word_new_song] 

        similar_words = []
        for focal_word in focal_words:
            if focal_word in model.wv.key_to_index:
                sim_words = model.wv.most_similar(positive=focal_word, topn=5)
                if sim_words:
                    similar_words.extend([(word, similarity) for word, similarity in sim_words])
                else:
                    similar_words.append(("No similar words found.", None))
            else:
                similar_words.append((focal_word + " not found in the model's vocabulary", None))
        
        norm_corpus = normalize_corpus(list(Billboard['Lyrics']))
        tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        tfidf_matrix = tf.fit_transform(norm_corpus)

        doc_sim = cosine_similarity(tfidf_matrix)
        doc_sim_df = pd.DataFrame(doc_sim)

        song_idx = len(Billboard) - 1
        song_similarities = doc_sim_df.iloc[song_idx].values
        similar_song_idxs = np.argsort(-song_similarities)[1:11]
        similar_songs = list(Billboard.iloc[similar_song_idxs]['Title'])
        similar_songs = [song for song in similar_songs if isinstance(song, str) and song.lower() != song_input]
        similar_songs = similar_songs[:5]

        return render_template_string(result_html, song_input=song_input, artist_input=artist_input,
                               word_count_new_song=word_count_new_song,
                               most_frequent_word_new_song=most_frequent_word_new_song,
                               similar_words=similar_words, similar_songs=similar_songs)

    else:
        return "Lyrics not found. Check for misspellings or try another song."


if __name__ == '__main__':
    app.run(debug=True)