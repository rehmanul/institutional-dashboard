# Institutional Persistence Dashboard

Production Streamlit application for **lobbying intelligence** and **policy persistence analysis**.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

---

## 🎯 Core Problem

> "Policy relevance does not persist across institutional change, forcing innovators to repeatedly re-translate the same capabilities across congressional sessions, venues, and vehicles."

---

## 🔧 Modules

| Module | Institutional Failure | Features |
|--------|----------------------|----------|
| **📝 Vocabulary Persistence** | Language doesn't persist | YAKE keywords, TF-IDF n-grams, vocabulary tracking |
| **⚠️ Institutional Framing** | Risk framing is opaque | VADER sentiment, NRC emotions, framing detection |
| **🎯 Issue Surfaces** | Coalition relevance invisible | BERTopic, zero-shot classification, jurisdiction overlap |
| **📊 Baseline Evidence** | Evidence is anecdotal | Descriptive stats, correlations, t-tests, ANOVA |
| **🔬 Analytical Robustness** | Action is unsafe | 14 estimators (OLS, Logit, Poisson, FE), LaTeX export |
| **🔗 Graph Export** | No structured output | Node/Edge schemas, Neo4j Cypher, lobbying intelligence |

---

## 🚀 Quick Start

### Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deploy to Render

1. Fork this repository
2. Connect to Render
3. Deploy using `render.yaml` blueprint

---

## 📦 Requirements

Core dependencies (minimal install):

```bash
pip install streamlit pandas plotly vaderSentiment yake statsmodels scipy scikit-learn altair wordcloud
```

Full install (with BERTopic for topic modeling):

```bash
pip install -r requirements.txt
```

---

## 🔗 Graph Export Schema

### Node Format (JSON-LD compatible)

```json
{
  "node_id": "119_S1234_101",
  "text": "The Secretary of Defense shall...",
  "bill_id": "S1234",
  "section": "101",
  "congress_session": 119,
  "sentiment": {
    "hostile": 0.75,
    "procedural": 0.10,
    "supportive": 0.05,
    "regulatory": 0.05,
    "neutral": 0.05
  },
  "entities": ["Department of Defense"],
  "keywords": ["supply chain", "procurement"]
}
```

### Edge Types

- `same_concept` - Semantic similarity between nodes
- `sentiment_change:calmed_hostility` - Policy posture shift
- `entity_flow` - Shared entity reference

### Neo4j Import

Export Cypher statements directly from the Graph Export page for database import.

---

## 📄 License

MIT
