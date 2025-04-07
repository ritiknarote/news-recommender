from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class NewsRecommender:
    def __init__(self, documents):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)

    def recommend(self, selected_index, top_n=5):
        cosine_similarities = cosine_similarity(
            self.tfidf_matrix[selected_index], self.tfidf_matrix
        ).flatten()
        related_docs_indices = cosine_similarities.argsort()[-top_n - 1:-1][::-1]
        return [(i, cosine_similarities[i]) for i in related_docs_indices]
