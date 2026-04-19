# Amharic AI Student Assistant
**Live Demo:** [Try the App](https://multilingual-app-chatbot-for-low-resource-languages.streamlit.app/)
## Problem Statement
Ethiopian university students often face challenges finding fast, accurate answers to questions regarding admissions, scholarships, fees, and requirements in both English and Amharic. This multilingual AI Student Assistant bridges that information gap by offering an accessible Streamlit interface that quickly retrieves relevant educational information.

## Dataset
The dataset contains 1010 Q&A pairs focused on topics such as:
- Admissions
- Exams
- Scholarships
- Study tips

Both English and Amharic entries are provided. Data is processed locally to yield fast similarity searches.

## Model Comparison
The application features two different retrieval systems for the sake of comparison and academic study:
1. **TF-IDF (Baseline)**: Computes word frequencies and inverse document frequencies using `sklearn.feature_extraction.text.TfidfVectorizer`. It relies on exact keyword matching.
2. **Sentence Embeddings (Modern)**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to capture semantic meaning. It calculates cosine similarity over the dense vectors, providing more context-aware results, especially when phrasing differs heavily from the dataset.

## UI Features
- **Interactive Chat Interface**: Familiar messaging bubble format.
- **Model Selection**: Evaluate TF-IDF, Embeddings, or compare both side by side.
- **Confidence Scoring**: High, Medium, or Low thresholds based on cosine similarity scores. Provides graceful fallback text on low confidence.
- **Top-K Results**: Expandable sections to view the top 3 closest matches.
- **Feedback Collection**: Built-in 👍 / 👎 buttons integrated directly in the chat to collect user feedback.
- **Language Auto-Detection**: Detects Amharic characters in the input and displays the detected language.
## Screenshots

### Chat Interface
(<img width="959" height="444" alt="image" src="https://github.com/user-attachments/assets/87f20d49-7583-416d-b8fe-d7a106e7ff8b" />
ng)

### Model Comparison (TF-IDF vs Embedding)
(<img width="943" height="434" alt="image" src="https://github.com/user-attachments/assets/33830080-5444-44f3-a747-e277e7b6df6f" />


### Confidence Scoring Example
(<img width="956" height="443" alt="image" src="https://github.com/user-attachments/assets/f027a242-aa2f-40cf-95f6-50f9814116b1" />

)
## Evaluation Logging
The web app includes an **Evaluation Mode** toggle. When enabled, user queries and model outputs/scores are logged locally to `evaluation/results.csv`, making it ideal for gathering user study metrics. User feedback (helpful/not helpful) is also logged in `evaluation/feedback.csv`.

## Limitations
- The embedding model (`all-MiniLM-L6-v2`) is primarily English-focused. While it may possess some cross-lingual capabilities, specialized Amharic embeddings would enhance native retrieval performance.
- Results rely strictly on the 1010 Q&A dictionary pairs. The system acts as a retriever and does not generate novel text via a language model (LLM).

## How to Run Locally
1. Configure a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install streamlit scikit-learn sentence-transformers pandas
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
