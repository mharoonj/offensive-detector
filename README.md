# Jigsaw Toxic Comment Classification Dataset to Qdrant

This project loads the Jigsaw Toxic Comment Classification Challenge dataset into a Qdrant vector database with emotion labels as metadata.

## Overview

The Jigsaw dataset contains Wikipedia comments that have been labeled for different types of toxicity:
- `toxic`: General toxic comments
- `severe_toxic`: Very toxic comments
- `obscene`: Obscene language
- `threat`: Threatening language
- `insult`: Insulting language
- `identity_hate`: Hate speech based on identity

This project maps these toxic categories to emotion labels and stores them in Qdrant for efficient similarity search and analysis.

## Features

- **Emotion Mapping**: Maps toxic categories to emotion labels (anger, fear, hate, disgust, etc.)
- **Vector Embeddings**: Uses sentence transformers for high-quality text embeddings
- **Batch Processing**: Efficiently processes large datasets in batches
- **Metadata Storage**: Stores comprehensive metadata including emotions, toxicity scores, and categories
- **Search Capabilities**: Enables semantic search and filtering by emotions

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Install and start Qdrant (if not already running):
```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant

# Or install locally
pip install qdrant
```

## Usage

### 1. Basic Loader (with dummy embeddings)

For quick testing or when you don't need high-quality embeddings:

```bash
python load_jigsaw_to_qdrant.py
```

### 2. Enhanced Loader (with proper embeddings)

For production use with high-quality sentence embeddings:

```bash
python load_jigsaw_to_qdrant_enhanced.py
```

### 3. Test and Explore the Data

After loading the data, you can test and explore it:

```bash
python test_jigsaw_qdrant.py
```

## Dataset Structure

The dataset should be organized as follows:
```
datasets/
└── jigsaw/
    ├── train.csv          # Training data with labels
    ├── test.csv           # Test data without labels
    ├── test_labels.csv    # Test data labels
    └── sample_submission.csv
```

## Emotion Mapping

The toxic categories are mapped to emotions as follows:

| Toxic Category | Emotions |
|----------------|----------|
| toxic | anger, disgust |
| severe_toxic | anger, disgust, fear |
| obscene | disgust, shame |
| threat | fear, anger |
| insult | anger, contempt |
| identity_hate | hate, anger, disgust |
| non-toxic | neutral |

## Metadata Structure

Each comment in Qdrant contains the following metadata:

```python
{
    'comment_text': 'Original comment text',
    'comment_id': 'Original dataset ID',
    'emotions': ['anger', 'disgust'],  # All detected emotions
    'primary_emotion': 'anger',        # Most prominent emotion
    'toxic_categories': ['toxic', 'insult'],  # Original toxic categories
    'toxicity_score': 0.33,            # Percentage of toxic categories (0-1)
    'is_toxic': 1,                     # Binary flags for each category
    'is_severe_toxic': 0,
    'is_obscene': 0,
    'is_threat': 0,
    'is_insult': 1,
    'is_identity_hate': 0,
    'total_toxic_labels': 2,           # Total number of toxic categories
    'data_split': 'train'              # train or test
}
```

## Search Examples

### Search by Emotion
```python
# Find comments with primary emotion 'anger'
tester.search_by_emotion('anger', limit=5)
```

### Semantic Search
```python
# Find comments similar to a query
tester.search_similar_comments("You are stupid", limit=5)
```

### Filter by Toxicity
```python
# Find high toxicity comments
tester.find_high_toxicity_comments(min_toxicity=0.5, limit=5)
```

## Performance Considerations

- **Batch Size**: Adjust batch size based on your memory constraints (default: 500 for enhanced, 1000 for basic)
- **Embedding Model**: The enhanced version uses `all-MiniLM-L6-v2` which provides good balance of speed and quality
- **Memory Usage**: Large datasets may require significant memory for processing
- **Qdrant Storage**: Ensure sufficient disk space for the vector database

## Troubleshooting

### Common Issues

1. **Qdrant Connection Error**: Make sure Qdrant is running on localhost:6333
2. **Memory Issues**: Reduce batch size in the loader scripts
3. **Missing Dependencies**: Install all requirements with `pip install -r requirements.txt`
4. **Dataset Not Found**: Ensure the dataset files are in the correct location

### Collection Management

```python
# List all collections
client = QdrantClient("localhost", port=6333)
collections = client.get_collections()

# Delete a collection (if needed)
client.delete_collection("jigsaw_toxic_comments_enhanced")
```

## Advanced Usage

### Custom Emotion Mapping

You can modify the emotion mappings in the `map_toxic_labels_to_emotions` method:

```python
emotion_mappings = {
    'toxic': ['anger', 'disgust'],
    'severe_toxic': ['anger', 'disgust', 'fear'],
    # ... customize as needed
}
```

### Different Embedding Models

Change the embedding model in the enhanced loader:

```python
loader = JigsawToQdrantLoaderEnhanced(embedding_model="all-mpnet-base-v2")
```

### Custom Search Filters

```python
# Search with multiple filters
search_filter = {
    "must": [
        {"key": "primary_emotion", "match": {"value": "anger"}},
        {"key": "toxicity_score", "range": {"gte": 0.5}}
    ]
}
```

## License

This project is for educational and research purposes. The Jigsaw dataset is provided by Google and is subject to their terms of use.

## Contributing

Feel free to submit issues and enhancement requests! 