import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from typing import List, Dict, Any
import logging
from sentence_transformers import SentenceTransformer
import torch
import time
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JigsawToQdrantLoaderEnhanced:
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333, 
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the enhanced Jigsaw to Qdrant loader.
        
        Args:
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            embedding_model: Sentence transformer model to use for embeddings
        """
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = "jigsaw_toxic_comments_enhanced"
        
        # Load sentence transformer model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
        
        logger.info(f"Embedding dimension: {self.vector_size}")
        
    def create_collection(self):
        """Create the Qdrant collection for storing toxic comments."""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )
            logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.info(f"Collection {self.collection_name} already exists or error: {e}")
    
    def get_optimized_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """
        Get optimized metadata with reduced attributes but keeping original labels unchanged.
        
        Args:
            row: DataFrame row with toxic labels
            
        Returns:
            Dictionary with optimized metadata
        """
        # Keep original labels exactly as they are in the dataset
        original_labels = {
            'toxic': row['toxic'],
            'severe_toxic': row['severe_toxic'],
            'obscene': row['obscene'],
            'threat': row['threat'],
            'insult': row['insult'],
            'identity_hate': row['identity_hate']
        }
        
        # Calculate primary emotion (simplified logic)
        if row['identity_hate'] == 1:
            primary_emotion = 'hate'
        elif row['threat'] == 1:
            primary_emotion = 'fear'
        elif row['obscene'] == 1:
            primary_emotion = 'disgust'
        elif row['severe_toxic'] == 1 or row['toxic'] == 1 or row['insult'] == 1:
            primary_emotion = 'anger'
        else:
            primary_emotion = 'neutral'
        
        # Calculate toxicity score
        toxic_count = sum(original_labels.values())
        toxicity_score = toxic_count / 6.0
        
        # Get toxic categories (only if present)
        toxic_categories = []
        if row['toxic'] == 1: toxic_categories.append('toxic')
        if row['severe_toxic'] == 1: toxic_categories.append('severe_toxic')
        if row['obscene'] == 1: toxic_categories.append('obscene')
        if row['threat'] == 1: toxic_categories.append('threat')
        if row['insult'] == 1: toxic_categories.append('insult')
        if row['identity_hate'] == 1: toxic_categories.append('identity_hate')
        
        return {
            'primary_emotion': primary_emotion,
            'toxicity_score': toxicity_score,
            'toxic_categories': toxic_categories,
            'is_toxic': int(toxic_count > 0),
            **original_labels  # Include original labels unchanged
        }
    
    def load_training_data(self, file_path: str, batch_size: int = 2000):
        """
        Load training data from CSV and insert into Qdrant with optimized processing.
        
        Args:
            file_path: Path to the training CSV file
            batch_size: Number of records to process in each batch
        """
        logger.info(f"Loading training data from {file_path} with batch size {batch_size}")
        
        # First, count total rows for progress bar
        logger.info("Counting total rows...")
        total_rows = sum(1 for _ in pd.read_csv(file_path, chunksize=batch_size))
        logger.info(f"Total rows to process: {total_rows}")
        
        # Read CSV in chunks to handle large files
        chunk_iter = pd.read_csv(file_path, chunksize=batch_size)
        
        total_processed = 0
        start_time = time.time()
        
        # Create progress bar for chunks
        with tqdm(total=total_rows, desc="Loading Training Data", unit="rows") as pbar:
            for chunk_num, chunk in enumerate(chunk_iter):
                # Prepare texts for batch embedding
                texts = chunk['comment_text'].tolist()
                
                # Create embeddings in batch
                embeddings = self.embedding_model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
                
                points = []
                
                for idx, (_, row) in enumerate(chunk.iterrows()):
                    try:
                        # Create optimized metadata
                        metadata = self.get_optimized_metadata(row)
                        metadata['comment_text'] = row['comment_text'][:200]  # Truncate text
                        metadata['comment_id'] = row['id']
                        metadata['data_split'] = 'train'
                        
                        # Create point
                        point = PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embeddings[idx].tolist(),
                            payload=metadata
                        )
                        
                        points.append(point)
                        
                    except Exception as e:
                        logger.error(f"Error processing row {row['id']}: {e}")
                        continue
                
                # Insert batch into Qdrant
                if points:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    total_processed += len(points)
                    
                    elapsed = time.time() - start_time
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    
                    # Update progress bar
                    pbar.update(len(chunk))
                    pbar.set_postfix({
                        'Processed': total_processed,
                        'Rate': f"{rate:.1f} rows/sec",
                        'Chunk': chunk_num + 1
                    })
        
        logger.info(f"Training data loading completed. Total processed: {total_processed}")
    
    def load_test_data(self, test_file: str, test_labels_file: str, batch_size: int = 2000):
        """
        Load test data and labels, then insert into Qdrant with optimized processing.
        
        Args:
            test_file: Path to the test CSV file
            test_labels_file: Path to the test labels CSV file
            batch_size: Number of records to process in each batch
        """
        logger.info(f"Loading test data from {test_file} and labels from {test_labels_file}")
        
        # Load test data and labels
        test_data = pd.read_csv(test_file)
        test_labels = pd.read_csv(test_labels_file)
        
        # Merge test data with labels
        test_merged = test_data.merge(test_labels, on='id', how='left')
        
        # Filter out unlabeled data
        test_merged = test_merged[~test_merged[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].eq(-1).any(axis=1)]
        
        logger.info(f"Filtered test data: {len(test_merged)} labeled samples")
        
        total_processed = 0
        start_time = time.time()
        
        # Process in batches with progress bar
        with tqdm(total=len(test_merged), desc="Loading Test Data", unit="rows") as pbar:
            for i in range(0, len(test_merged), batch_size):
                batch = test_merged.iloc[i:i+batch_size]
                
                # Prepare texts for batch embedding
                texts = batch['comment_text'].tolist()
                
                # Create embeddings in batch
                embeddings = self.embedding_model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
                
                points = []
                
                for idx, (_, row) in enumerate(batch.iterrows()):
                    try:
                        # Create optimized metadata
                        metadata = self.get_optimized_metadata(row)
                        metadata['comment_text'] = row['comment_text'][:200]  # Truncate text
                        metadata['comment_id'] = row['id']
                        metadata['data_split'] = 'test'
                        
                        # Create point
                        point = PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embeddings[idx].tolist(),
                            payload=metadata
                        )
                        
                        points.append(point)
                        
                    except Exception as e:
                        logger.error(f"Error processing test row {row['id']}: {e}")
                        continue
                
                # Insert batch into Qdrant
                if points:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    total_processed += len(points)
                    
                    elapsed = time.time() - start_time
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    
                    # Update progress bar
                    pbar.update(len(batch))
                    pbar.set_postfix({
                        'Processed': total_processed,
                        'Rate': f"{rate:.1f} rows/sec",
                        'Batch': i//batch_size + 1
                    })
        
        logger.info(f"Test data loading completed. Total processed: {total_processed}")
    
    def get_collection_info(self):
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            logger.info(f"Collection info: {info}")
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
    
    def search_examples(self, query_text: str, limit: int = 5, filter_emotion: str = None):
        """
        Search for similar comments in the collection.
        
        Args:
            query_text: Text to search for
            limit: Number of results to return
            filter_emotion: Optional emotion filter
        """
        try:
            query_embedding = self.embedding_model.encode(query_text, convert_to_tensor=False)
            
            # Build search filter
            search_filter = None
            if filter_emotion:
                search_filter = {
                    "must": [
                        {
                            "key": "primary_emotion",
                            "match": {"value": filter_emotion}
                        }
                    ]
                }
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit,
                query_filter=search_filter
            )
            
            logger.info(f"Search results for '{query_text}':")
            for i, result in enumerate(results):
                logger.info(f"Result {i+1}:")
                logger.info(f"  Score: {result.score:.3f}")
                logger.info(f"  Text: {result.payload['comment_text']}")
                logger.info(f"  Primary Emotion: {result.payload['primary_emotion']}")
                logger.info(f"  Toxic Categories: {result.payload['toxic_categories']}")
                logger.info(f"  Toxicity Score: {result.payload['toxicity_score']:.2f}")
                logger.info(f"  Original Labels: toxic={result.payload['toxic']}, severe_toxic={result.payload['severe_toxic']}, "
                          f"obscene={result.payload['obscene']}, threat={result.payload['threat']}, "
                          f"insult={result.payload['insult']}, identity_hate={result.payload['identity_hate']}")
                logger.info("---")
                
        except Exception as e:
            logger.error(f"Error searching: {e}")
    
    def get_label_statistics(self):
        """Get statistics about toxic labels in the collection."""
        try:
            logger.info("Getting collection statistics...")
            
            # Get all points to analyze labels
            all_points = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust based on your data size
                with_payload=True,
                with_vectors=False
            )[0]
            
            label_counts = {
                'toxic': 0,
                'severe_toxic': 0,
                'obscene': 0,
                'threat': 0,
                'insult': 0,
                'identity_hate': 0,
                'neutral': 0
            }
            
            emotion_counts = {}
            
            # Process with progress bar
            with tqdm(total=len(all_points), desc="Analyzing Statistics", unit="points") as pbar:
                for point in all_points:
                    # Count toxic labels (using original values)
                    for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
                        if point.payload.get(label, 0) == 1:
                            label_counts[label] += 1
                    
                    # Count emotions
                    emotion = point.payload.get('primary_emotion', 'unknown')
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    
                    # Count neutral (no toxic labels)
                    if sum(point.payload.get(label, 0) for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']) == 0:
                        label_counts['neutral'] += 1
                    
                    pbar.update(1)
            
            logger.info("Original Toxic Label Statistics:")
            for label, count in label_counts.items():
                logger.info(f"  {label}: {count}")
            
            logger.info("\nEmotion Statistics:")
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {emotion}: {count}")
                
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")

def main():
    """Main function to load Jigsaw dataset into Qdrant."""
    
    start_time = time.time()
    
    # Initialize loader
    loader = JigsawToQdrantLoaderEnhanced()
    
    # Create collection
    loader.create_collection()
    
    # Load training data with optimized settings
    logger.info("Starting optimized training data load...")
    loader.load_training_data("datasets/jigsaw/train.csv", batch_size=2000)
    
    # Load test data with optimized settings
    # logger.info("Starting optimized test data load...")
    # loader.load_test_data("datasets/jigsaw/test.csv", "datasets/jigsaw/test_labels.csv", batch_size=2000)
    
    # Get collection information
    loader.get_collection_info()
    
    # Get label statistics
    loader.get_label_statistics()
    
    # Example searches
    logger.info("Performing example searches...")
    loader.search_examples("You are stupid and worthless", limit=3)
    loader.search_examples("Hello, how are you today?", limit=3)
    loader.search_examples("I hate you", limit=3, filter_emotion="hate")
    
    total_time = time.time() - start_time
    logger.info(f"Total processing time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main() 