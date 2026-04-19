import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingModel:
    def __init__(self, data_path='data/dataset.json', data=None):
        self.data_path = data_path
        # Load the sentence transformer model (updated for multilingual capacity)
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        if data is not None:
            self.data = data
        else:
            self.data = self._load_data()
            
        # Prepare corpus and precompute embeddings
        self.corpus = [item['question'] for item in self.data]
        self.embeddings = self.model.encode(self.corpus)
        
    def _load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def retrieve(self, query, k=3):
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
        
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
