# LinkedIn Profile Search Dashboard

A powerful web-based dashboard that allows users to upload student or individual name/university data and automatically search for potential LinkedIn matches. It uses advanced semantic similarity via MPNet, fuzzy matching, SerpAPI queries, and Selenium-based Bing fallbacks. Results include estimated location and income, when available.

---

## 🚀 Features

- Upload a CSV of names and universities, or enter data manually.
- Multi-threaded search using SerpAPI and Bing.
- Intelligent matching with SentenceTransformer and RapidFuzz.
- Auto-detection of graduation year, estimated location, and income.
- Confidence scoring using cosine similarity and fuzzy ratios.
- Clean UI with progress tracking and CSV export.

---

## 📦 Requirements

Install the necessary Python packages using:

```bash
pip install pandas dash chardet selenium requests sentence-transformers rapidfuzz webdriver-manager numpy
```

You will also need:

- Chrome installed (for Selenium headless browser)
- [SerpAPI](https://serpapi.com/) key (you can add your free API key in the code or via environment variable)

---

## 📁 How to Run

```bash
python your_script_name.py
```

A Chrome app window will open with the dashboard running at `http://127.0.0.1:8050/`.

---

## 📌 Notes

- Free SerpAPI accounts are limited to 100 queries/month. The app will automatically fall back to Bing scraping after that.
- Locations and income are estimated based on keywords and may not always be accurate.
- Manual mode lets you search a single person and enables a "🔎 Manual Mode" indicator.
- CSV format should include columns like: `First Name`, `Last Name`, `University`, `Graduation Year` (optional).

---

## 👥 Team Members

This project was created entirely by:

- Arik Gitman  
- Soham Desai  
- Andrew Jerome  
- Dylan Shah  
- Darsh Patel

---

## 🛠️ TODO / Future Work

- Add GitHub integration and auto-deployment.
- Improve accuracy of location estimation.
- Add LinkedIn scraping if logged-in cookies are provided.
- Use OpenAI embeddings or LLMs for fallback summaries.
