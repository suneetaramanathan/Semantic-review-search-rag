# Semantic Review Search, Re-Ranking & RAG

End-to-end search system over 5,000 customer reviews: semantic retrieval, cross-encoder re-ranking,
gradient-boosting baselines and an LLM summary (RAG), served via FastAPI.

## Pipeline
1. **Retrieval:** sentence embeddings (all-MiniLM-L6-v2, 384-d), cosine similarity search
2. **Re-ranking:** cross-encoder (ms-marco-MiniLM-L-6-v2) re-scores the top 50 candidates
3. **Evaluation:** Precision@10 and MRR on sentiment-labelled queries
4. **Gradient boosting:** LightGBM on TF-IDF and on embeddings, compared with fine-tuned DistilBERT
5. **RAG:** Qwen2.5-0.5B-Instruct summarises the top-ranked reviews
6. **API:** FastAPI `/search` endpoint returning the summary and ranked results

## Results
| Search | Precision@10 | MRR |
|---|---|---|
| Bi-encoder only | 0.940 | 1.000 |
| + Cross-encoder re-ranking | 0.950 | 1.000 |

| Classifier (same 1,000 test reviews) | Accuracy | F1 |
|---|---|---|
| LightGBM + TF-IDF | 0.889 | 0.886 |
| LightGBM + embeddings | 0.866 | 0.863 |
| Fine-tuned DistilBERT (PyTorch) | 0.913 | 0.905 |

## Learnings
- Embeddings capture topic more than polarity ("friendly staff" can retrieve "rude staff").
- Re-ranking improves precision; MRR was already at its ceiling.
- LightGBM + TF-IDF is a strong, cheap and interpretable baseline.
- Small evaluation set (10 queries): results are directional, not conclusive.

Related project: [Review Sentiment Classifier - PyTorch + FastAPI](https://github.com/suneetaramanathan/Review-sentiment-pytorch-fastapi)
