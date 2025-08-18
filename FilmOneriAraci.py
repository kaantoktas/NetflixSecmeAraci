import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time 
from config import TMDB_API_KEY 

BASE_URL = "https://api.themoviedb.org/3"

def fetch_data_with_retry(url, params=None, max_retries=5, initial_delay=1):
    for i in range(max_retries):
        try:
            response = requests.get(url, params=params)
            response.raise_for_status() 
            return response.json()
        except requests.exceptions.RequestException as e:
            if i < max_retries - 1:
                delay = initial_delay * (2 ** i)
                print(f"İstek başarısız oldu ({e}). {delay} saniye sonra tekrar deniyorum...")
                time.sleep(delay)
            else:
                print(f"Maksimum deneme sayısı aşıldı. İstek başarısız oldu: {e}")
                return None

def search_movie(query):
    params = {"api_key": TMDB_API_KEY, "query": query, "language": "tr-TR"} 
    url = f"{BASE_URL}/search/movie"
    data = fetch_data_with_retry(url, params)

    if data and data['results']:
        first_result = data['results'][0]
        return first_result['id'], first_result['title']
    return None, None

def get_movie_details(movie_id):
    params = {"api_key": TMDB_API_KEY, "language": "tr-TR"} 
    url = f"{BASE_URL}/movie/{movie_id}"
    data = fetch_data_with_retry(url, params)
    return data

def get_movie_credits(movie_id):
    params = {"api_key": TMDB_API_KEY, "language": "tr-TR"} 
    url = f"{BASE_URL}/movie/{movie_id}/credits"
    data = fetch_data_with_retry(url, params)
    return data

def get_movie_keywords(movie_id):
    params = {"api_key": TMDB_API_KEY} 
    url = f"{BASE_URL}/movie/{movie_id}/keywords"
    data = fetch_data_with_retry(url, params)
    return data

def extract_features(movie_details, movie_credits, movie_keywords):
    genres = [genre['name'] for genre in movie_details.get('genres', [])]
    
    cast = [c['name'] for c in movie_credits.get('cast', [])[:3]]
    
    director = [c['name'] for c in movie_credits.get('crew', []) if c['job'] == 'Director']
    director = director[0] if director else ''

    keywords = [kw['name'] for kw in movie_keywords.get('keywords', [])]

    features = []
    features.extend([g.lower().replace(" ", "") for g in genres])
    features.extend([c.lower().replace(" ", "") for c in cast])
    if director:
        features.append(director.lower().replace(" ", ""))
    features.extend([kw.lower().replace(" ", "") for kw in keywords])
    
    return " ".join(features)

def get_recommendations(favorite_movie_title, num_recommendations=5, max_search_results=100):
    favorite_movie_id, favorite_movie_name = search_movie(favorite_movie_title)
    if not favorite_movie_id:
        print(f"Üzgünüm, '{favorite_movie_title}' filmini bulamadım. Lütfen doğru yazdığınızdan emin olun.")
        return []

    print(f"Favori filminiz '{favorite_movie_name}' olarak bulundu. Benzer filmler aranıyor...")

    fav_details = get_movie_details(favorite_movie_id)
    fav_credits = get_movie_credits(favorite_movie_id)
    fav_keywords = get_movie_keywords(favorite_movie_id)

    if not fav_details or not fav_credits or not fav_keywords:
        print(f"'{favorite_movie_name}' filminin detayları çekilemedi. Öneri yapılamıyor.")
        return []

    favorite_features = extract_features(fav_details, fav_credits, fav_keywords)
    
    if not favorite_features:
        print(f"'{favorite_movie_name}' filmi için yeterli özellik bulunamadı. Öneri yapılamıyor.")
        return []

    popular_movies_url = f"{BASE_URL}/movie/popular"
    all_movies = []
    page = 1
    while len(all_movies) < max_search_results and page <= 500 // 20: 
        params = {"api_key": TMDB_API_KEY, "page": page, "language": "tr-TR"}
        popular_data = fetch_data_with_retry(popular_movies_url, params)
        if popular_data and popular_data['results']:
            all_movies.extend(popular_data['results'])
            page += 1
            time.sleep(0.1) 
        else:
            break
    
    all_movies_filtered = [m for m in all_movies if m['id'] != favorite_movie_id]

    if not all_movies_filtered:
        print("Yeterli sayıda film bulunamadı. Lütfen daha sonra tekrar deneyin veya daha popüler bir film girin.")
        return []

    movie_data = []
    print(f"{len(all_movies_filtered)} adet filmin detayları çekiliyor...")
    for movie in all_movies_filtered:
        movie_id = movie['id']
        details = get_movie_details(movie_id)
        credits = get_movie_credits(movie_id)
        keywords = get_movie_keywords(movie_id)

        if details and credits and keywords:
            features = extract_features(details, credits, keywords)
            if features:
                movie_data.append({
                    'id': movie_id,
                    'title': movie['title'],
                    'features': features
                })
        time.sleep(0.05) 

    df = pd.DataFrame(movie_data)
    
    if df.empty:
        print("Özellikleri çıkarılacak yeterli film bulunamadı. Öneri yapılamıyor.")
        return []

    df_fav = pd.DataFrame([{'id': favorite_movie_id, 'title': favorite_movie_name, 'features': favorite_features}])
    df = pd.concat([df_fav, df], ignore_index=True)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english') 
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['features'])

    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)

    sim_scores = list(enumerate(cosine_sim[0]))[1:] 

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommended_indices = [i[0] for i in sim_scores[:num_recommendations]]

    return df['title'].iloc[recommended_indices].tolist()

if __name__ == "__main__":
    print("Film Öneri Programına Hoş Geldiniz!")
    print("Lütfen favori bir filminizin adını girin (örn: Inception, The Matrix, Interstellar).")
    
    while True:
        user_input = input("\nFavori filminizin adı (Çıkmak için 'q' tuşuna basın): ").strip()
        if user_input.lower() == 'q':
            print("Programdan çıkılıyor. Hoşça kalın!")
            break
        
        if not user_input:
            print("Lütfen bir film adı girin.")
            continue

        recommendations = get_recommendations(user_input)

        if recommendations:
            print(f"\n'{user_input}' filmine benzer öneriler:")
            for i, movie in enumerate(recommendations):
                print(f"{i+1}. {movie}")
        else:
            print("Öneri yapılamadı veya girilen film bulunamadı. Lütfen farklı bir film deneyin.")
