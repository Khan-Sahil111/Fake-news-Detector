# 📰 Fake News Detector

This project is a **Fake News Detection System** that classifies news articles as either **Real** or **Fake** using Natural Language Processing (NLP) and a Logistic Regression model. The model is trained on publicly available datasets and includes a user-friendly **Streamlit web app** for real-time predictions.

---

## 📁 Project Structure

Fake-News-Detector/
│
├── Fake.csv # Dataset of fake news articles
├── True.csv # Dataset of real news articles
├── Fake news Detector.ipynb # Jupyter notebook for model training and evaluation
├── app.py # Streamlit app for user interaction
├── model.pkl # Trained Logistic Regression model
├── vectorizer.pkl # Fitted TF-IDF vectorizer
├── lime_explanation.html # LIME model explanation visualization
└── README.md # Project documentation

yaml
Copy
Edit

---

## 🧠 Model Overview

- **Algorithm**: Logistic Regression  
- **Vectorization**: TF-IDF (max 5000 features)  
- **Text Preprocessing**:
  - Lowercasing
  - Removing punctuation
  - Stopword removal
  - Lemmatization (using WordNet)

---

## 📊 Evaluation Metrics

- Accuracy, Precision, Recall, and F1-Score
- Evaluated using an 80/20 train-test split
- LIME (Local Interpretable Model-agnostic Explanations) used for interpretability

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
2. Install dependencies
It's recommended to use a virtual environment:

bash
Copy
Edit
pip install -r requirements.txt
If requirements.txt doesn't exist, install manually:

bash
Copy
Edit
pip install pandas scikit-learn streamlit nltk joblib lime
3. Download NLTK data
These are required for stopword removal and lemmatization:

python
Copy
Edit
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
4. Run the Streamlit App
bash
Copy
Edit
streamlit run app.py
🧪 How It Works
User Input: Paste or type a news article into the text area.

Processing: The text is cleaned and transformed using the TF-IDF vectorizer.

Prediction: The logistic regression model predicts whether the article is real or fake.

Output: Displays the result along with the model’s confidence.

🔍 LIME Explanation
This project includes lime_explanation.html to visually explain how the model makes decisions on specific predictions using the LIME framework.

To view the explanation:

bash
Copy
Edit
open lime_explanation.html
# or just double-click the file to open it in a browser
📦 Model Files
model.pkl: The trained logistic regression classifier

vectorizer.pkl: The TF-IDF vectorizer fitted on the cleaned dataset

These are used directly by the Streamlit app for predictions.

📚 Datasets
Fake.csv – A dataset of fake news articles

True.csv – A dataset of real news articles

These are concatenated, labeled, shuffled, and used to train the model.

✅ Future Improvements
Add deep learning-based models (e.g., LSTM, BERT)

Collect and clean a more recent dataset

Add deployment via platforms like Streamlit Cloud or Heroku

Add more robust evaluation (e.g., cross-validation)

🤝 Contributions
Contributions, suggestions, and pull requests are welcome! Feel free to fork the repository and improve the project.

📃 License
This project is open-source and available under the MIT License.

✍️ Author
SAHIL
