import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFModel:
    def __init__(self, data_path='data/dataset.json', data=None):
        self.data_path = data_path
        self.vectorizer = TfidfVectorizer()
        
        if data is not None:
            self.data = data
        else:
            self.data = self._load_data()
            
        # Prepare corpus from all questions
        self.corpus = [item['question'] for item in self.data]
        # Fit vectorizer and compute tf-idf matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        
    def _load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def retrieve(self, query, k=3):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top-k indices, sorted decending
        top_indices = similarities.argsort()[-k:][::-1]
        
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score > 0.8:
                confidence = "High"
            elif score >= 0.5:
                confidence = "Medium"
            else:
                confidence = "Low"
                
            results.append({
                "question": self.data[idx]['question'],
                "answer": self.data[idx]['answer'],
                "language": self.data[idx]['language'],
                "score": float(score),
                "confidence": confidence
            })
            
        return results
