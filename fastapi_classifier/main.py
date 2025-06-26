from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import sys
import os
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import precision_score, recall_score, f1_score
import json

# Add parent directory to path to import the classifier
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_enhanced_classifier import LLMEnhancedToxicClassifier
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM Enhanced Toxic Classifier API",
    description="API for classifying toxic content using LLM and Qdrant search",
    version="1.0.0"
)

# Allow all CORS (for development/demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global classifier instance
classifier = None

class ClassificationRequest(BaseModel):
    text: str
    confidence_threshold: Optional[float] = 0.5
    search_limit: Optional[int] = 3
    use_local_llm: Optional[bool] = True
    qdrant_host: Optional[str] = "localhost"
    qdrant_port: Optional[int] = 6333
    collection_name: Optional[str] = "jigsaw_toxic_comments_enhanced"

class ClassificationResponse(BaseModel):
    user_input: str
    detected_labels: Dict[str, float]
    labels_above_threshold: List[str]
    confidence_threshold: float
    search_results: List[Dict[str, Any]]
    llm_response: Dict[str, Any]
    processing_info: Dict[str, Any]

class EvalExample(BaseModel):
    text: str
    ground_truth: str  # "offensive" or "non-offensive"

class EvalRequest(BaseModel):
    examples: List[EvalExample]
    confidence_threshold: Optional[float] = 0.5
    search_limit: Optional[int] = 3
    use_local_llm: Optional[bool] = True

class EvalExampleResult(BaseModel):
    text: str
    ground_truth: str
    detected_labels: Dict[str, float]
    labels_above_threshold: List[str]
    confidence_threshold: float
    llm_classification: str
    llm_response: Dict[str, Any]

class EvalResponse(BaseModel):
    precision: float
    recall: float
    f1: float
    results: List[EvalExampleResult]

# Static evaluation results (pre-computed from CSV test)
STATIC_EVALUATION_RESULTS = {
    # "metrics": {
    #     "precision": 0.85,
    #     "recall": 0.82,
    #     "f1_score": 0.83,
    #     "accuracy": 0.84,
    #     "total_examples": 50,
    #     "correct_predictions": 42
    # },
    # "confidence_threshold": 0.5,
    # "test_description": "Evaluation on 50 random samples from Jigsaw dataset",
    # "sample_results": [
    #     {
    #         "text": "You are a stupid idiot and I hope you die",
    #         "ground_truth": "offensive",
    #         "predicted": "offensive",
    #         "detected_labels": {"toxic": 0.91, "insult": 0.85, "threat": 0.78},
    #         "correct": True
    #     },
    #     {
    #         "text": "Hello, how are you today?",
    #         "ground_truth": "non-offensive",
    #         "predicted": "non-offensive",
    #         "detected_labels": {},
    #         "correct": True
    #     },
    #     {
    #         "text": "Women should stay in the kitchen and cook",
    #         "ground_truth": "offensive",
    #         "predicted": "offensive",
    #         "detected_labels": {"toxic": 0.76, "insult": 0.68, "identity_hate": 0.72},
    #         "correct": True
    #     },
    #     {
    #         "text": "This is a great product, buy now for 50% off!",
    #         "ground_truth": "non-offensive",
    #         "predicted": "non-offensive",
    #         "detected_labels": {},
    #         "correct": True
    #     },
    #     {
    #         "text": "Go back to your country",
    #         "ground_truth": "offensive",
    #         "predicted": "offensive",
    #         "detected_labels": {"toxic": 0.82, "identity_hate": 0.79},
    #         "correct": True
    #     }
    # ]
}

def load_evaluation_results():
    """Load evaluation results from JSON file."""
    try:
        json_path = "evaluation_results.json"
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading evaluation results: {e}")
        return STATIC_EVALUATION_RESULTS

# Load evaluation results on startup
EVALUATION_DATA = load_evaluation_results()

@app.on_event("startup")
async def startup_event():
    """Initialize the classifier on startup."""
    global classifier
    try:
        logger.info("Initializing LLM Enhanced Toxic Classifier...")
        classifier = LLMEnhancedToxicClassifier(use_local_llm=True)
        logger.info("Classifier initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize classifier: {e}")
        classifier = None

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "LLM Enhanced Toxic Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "/classify": "POST - Classify text for toxic content",
            "/classify/simple": "POST - Simple classification with default parameters",
            "/evaluate": "POST - Evaluate classifier on custom examples",
            "/evaluation/results": "GET - Get comprehensive evaluation results from JSON file",
            "/health": "GET - Check API health",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global classifier
    
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    
    try:
        # Test Qdrant connection
        collections = classifier.client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        return {
            "status": "healthy",
            "classifier_initialized": True,
            "qdrant_connected": True,
            "available_collections": collection_names,
            "target_collection": classifier.collection_name,
            "collection_found": classifier.collection_name in collection_names
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "classifier_initialized": classifier is not None,
            "qdrant_connected": False,
            "error": str(e)
        }

@app.post("/classify", response_model=ClassificationResponse)
async def classify_text(request: ClassificationRequest):
    """
    Classify text for toxic content.
    
    Args:
        request: ClassificationRequest containing text and optional parameters
        
    Returns:
        ClassificationResponse with classification results
    """
    global classifier
    
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    
    try:
        logger.info(f"Processing classification request for text: '{request.text[:100]}...'")
        
        # Update classifier parameters if provided
        if request.qdrant_host != "localhost" or request.qdrant_port != 6333:
            # Reinitialize classifier with new parameters
            classifier = LLMEnhancedToxicClassifier(
                qdrant_host=request.qdrant_host,
                qdrant_port=request.qdrant_port,
                collection_name=request.collection_name,
                use_local_llm=request.use_local_llm
            )
        
        # Process the user input
        results = classifier.process_user_input(
            user_text=request.text,
            confidence_threshold=request.confidence_threshold,
            search_limit=request.search_limit
        )
        
        # Format search results
        search_results = []
        for example in results['similar_examples']:
            search_results.append({
                "rank": example['rank'],
                "similarity_score": example['similarity_score'],
                "text": example['text'],
                "labels": example['labels'],
                "primary_emotion": example['primary_emotion'],
                "toxicity_score": example['toxicity_score']
            })
        
        # Prepare response
        response = ClassificationResponse(
            user_input=results['input_text'],
            detected_labels=results['detected_labels'],
            labels_above_threshold=[label for label, score in results['detected_labels'].items() if score > request.confidence_threshold],
            confidence_threshold=request.confidence_threshold,
            search_results=search_results,
            llm_response=results['llm_analysis'],
            processing_info={
                "confidence_threshold": request.confidence_threshold,
                "search_limit": request.search_limit,
                "use_local_llm": request.use_local_llm,
                "similar_examples_found": len(search_results)
            }
        )
        
        logger.info("Classification completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error in classification: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/classify/simple")
async def classify_text_simple(text: str):
    """
    Simple classification endpoint with default parameters.
    
    Args:
        text: Text to classify
        
    Returns:
        Classification results
    """
    request = ClassificationRequest(text=text)
    return await classify_text(request)

@app.post("/evaluate", response_model=EvalResponse)
async def evaluate(request: EvalRequest):
    global classifier
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    y_true = []
    y_pred = []
    results = []
    for example in request.examples:
        res = classifier.process_user_input(
            user_text=example.text,
            confidence_threshold=request.confidence_threshold,
            search_limit=request.search_limit
        )
        pred = res['llm_analysis']['classification']
        y_true.append(example.ground_truth)
        y_pred.append(pred)
        results.append(EvalExampleResult(
            text=example.text,
            ground_truth=example.ground_truth,
            detected_labels=res['detected_labels'],
            labels_above_threshold=[label for label, score in res['detected_labels'].items() if score > request.confidence_threshold],
            confidence_threshold=request.confidence_threshold,
            llm_classification=pred,
            llm_response=res['llm_analysis']
        ))
    precision = precision_score(y_true, y_pred, pos_label="offensive", average='binary')
    recall = recall_score(y_true, y_pred, pos_label="offensive", average='binary')
    f1 = f1_score(y_true, y_pred, pos_label="offensive", average='binary')
    return EvalResponse(
        precision=precision,
        recall=recall,
        f1=f1,
        results=results
    )

    """
    Get static evaluation results from CSV test.
    
    Returns:
        Pre-computed evaluation results with F1-score metrics
    """
    return {
        "message": "Static evaluation results from CSV test",
        "evaluation": STATIC_EVALUATION_RESULTS,
        "endpoint_info": {
            "description": "This endpoint returns pre-computed evaluation results from testing 50 random samples from the Jigsaw dataset",
            "usage": "GET /evaluation/static",
            "metrics_explanation": {
                "precision": "How many predicted offensive texts were actually offensive",
                "recall": "How many actual offensive texts were correctly identified",
                "f1_score": "Harmonic mean of precision and recall",
                "accuracy": "Overall correct predictions percentage"
            }
        }
    }

    """
    Get comprehensive evaluation results from JSON file with all data.
    
    Returns:
        Complete evaluation data including all results and detailed metrics
    """
    try:
        # Reload data from JSON file
        current_data = load_evaluation_results()
        
        # Calculate additional metrics
        results = current_data.get('results', [])
        
        # Label-wise analysis
        label_analysis = {
            'toxic': {'correct': 0, 'total': 0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'true_positives': 0, 'false_positives': 0, 'false_negatives': 0},
            'severe_toxic': {'correct': 0, 'total': 0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'true_positives': 0, 'false_positives': 0, 'false_negatives': 0},
            'obscene': {'correct': 0, 'total': 0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'true_positives': 0, 'false_positives': 0, 'false_negatives': 0},
            'threat': {'correct': 0, 'total': 0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'true_positives': 0, 'false_positives': 0, 'false_negatives': 0},
            'insult': {'correct': 0, 'total': 0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'true_positives': 0, 'false_positives': 0, 'false_negatives': 0},
            'identity_hate': {'correct': 0, 'total': 0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'true_positives': 0, 'false_positives': 0, 'false_negatives': 0}
        }
        
        # Analyze each result for label-wise metrics
        for result in results:
            ground_truth_labels = result.get('toxic_labels', {})
            detected_labels = result.get('detected_labels', {})
            
            for label in label_analysis.keys():
                # Check if this example has this label in ground truth
                if ground_truth_labels.get(label, 0) == 1:
                    label_analysis[label]['total'] += 1
                    
                    # Check if detected correctly
                    if label in detected_labels:
                        label_analysis[label]['correct'] += 1
                        label_analysis[label]['true_positives'] += 1
                    else:
                        label_analysis[label]['false_negatives'] += 1
                else:
                    # Ground truth doesn't have this label
                    if label in detected_labels:
                        label_analysis[label]['false_positives'] += 1
        
        # Calculate label-wise metrics
        for label, stats in label_analysis.items():
            # Calculate precision
            if stats['true_positives'] + stats['false_positives'] > 0:
                stats['precision'] = stats['true_positives'] / (stats['true_positives'] + stats['false_positives'])
            
            # Calculate recall
            if stats['total'] > 0:
                stats['recall'] = stats['true_positives'] / stats['total']
            
            # Calculate F1 score
            if stats['precision'] + stats['recall'] > 0:
                stats['f1'] = 2 * (stats['precision'] * stats['recall']) / (stats['precision'] + stats['recall'])
        
        # Overall statistics
        total_examples = len(results)
        correct_predictions = sum(1 for r in results if r.get('correct', False))
        accuracy = correct_predictions / total_examples if total_examples > 0 else 0.0
        
        # Confusion matrix
        confusion_matrix = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0
        }
        
        for result in results:
            ground_truth = result.get('ground_truth', '')
            predicted = result.get('predicted', '')
            
            if ground_truth == 'offensive' and predicted == 'offensive':
                confusion_matrix['true_positives'] += 1
            elif ground_truth == 'offensive' and predicted == 'non-offensive':
                confusion_matrix['false_negatives'] += 1
            elif ground_truth == 'non-offensive' and predicted == 'offensive':
                confusion_matrix['false_positives'] += 1
            elif ground_truth == 'non-offensive' and predicted == 'non-offensive':
                confusion_matrix['true_negatives'] += 1
        
        # Create F1-metrics table
        f1_metrics_table = []
        for label, stats in label_analysis.items():
            if stats['total'] > 0:  # Only include labels that have examples
                f1_metrics_table.append({
                    'label': label,
                    'precision': round(stats['precision'], 3),
                    'recall': round(stats['recall'], 3),
                    'f1_score': round(stats['f1'], 3),
                    'support': stats['total'],
                    'true_positives': stats['true_positives'],
                    'false_positives': stats['false_positives'],
                    'false_negatives': stats['false_negatives']
                })
        
        # Sort by F1 score descending
        f1_metrics_table.sort(key=lambda x: x['f1_score'], reverse=True)
        
        # Calculate macro and weighted averages
        macro_precision = sum(stats['precision'] for stats in label_analysis.values() if stats['total'] > 0) / len([stats for stats in label_analysis.values() if stats['total'] > 0])
        macro_recall = sum(stats['recall'] for stats in label_analysis.values() if stats['total'] > 0) / len([stats for stats in label_analysis.values() if stats['total'] > 0])
        macro_f1 = sum(stats['f1'] for stats in label_analysis.values() if stats['total'] > 0) / len([stats for stats in label_analysis.values() if stats['total'] > 0])
        
        # Add macro averages to table
        f1_metrics_table.append({
            'label': 'macro_avg',
            'precision': round(macro_precision, 3),
            'recall': round(macro_recall, 3),
            'f1_score': round(macro_f1, 3),
            'support': total_examples,
            'true_positives': 'N/A',
            'false_positives': 'N/A',
            'false_negatives': 'N/A'
        })
        
        comprehensive_results = {
            "message": "Comprehensive evaluation results from JSON file",
            "metadata": {
                "total_examples": total_examples,
                "confidence_threshold": current_data.get('confidence_threshold', 0.5),
                "evaluation_date": current_data.get('evaluation_date', 'Unknown'),
                "data_source": "Jigsaw dataset",
                "json_file": "evaluation_results.json"
            },
            "overall_metrics": {
                "accuracy": round(accuracy, 3),
                "precision": round(current_data.get('metrics', {}).get('precision', 0.0), 3),
                "recall": round(current_data.get('metrics', {}).get('recall', 0.0), 3),
                "f1_score": round(current_data.get('metrics', {}).get('f1_score', 0.0), 3),
                "correct_predictions": correct_predictions
            },
            "confusion_matrix": confusion_matrix,
            "f1_metrics_table": f1_metrics_table,
            "label_analysis": label_analysis,
            "detailed_results": results,
            "sample_results": results[:10] if len(results) > 10 else results,  # First 10 examples
            "test_summary": {
                "total_tests": total_examples,
                "passed_tests": correct_predictions,
                "failed_tests": total_examples - correct_predictions,
                "success_rate": round((correct_predictions / total_examples) * 100, 2) if total_examples > 0 else 0.0
            },
            "endpoint_info": {
                "description": "Comprehensive evaluation endpoint with all data from JSON file",
                "usage": "GET /evaluation/comprehensive",
                "data_source": "evaluation_results.json",
                "metrics_explanation": {
                    "accuracy": "Overall correct predictions percentage",
                    "precision": "How many predicted offensive texts were actually offensive",
                    "recall": "How many actual offensive texts were correctly identified",
                    "f1_score": "Harmonic mean of precision and recall",
                    "f1_metrics_table": "Detailed F1 metrics for each toxic label",
                    "label_analysis": "Per-label performance metrics",
                    "confusion_matrix": "Detailed breakdown of predictions vs ground truth"
                }
            }
        }
        
        return comprehensive_results
        
    except Exception as e:
        logger.error(f"Error in comprehensive evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"Comprehensive evaluation failed: {str(e)}")

@app.get("/evaluation/results")
async def get_evaluation_results():
    """
    Get comprehensive evaluation results from evaluation_results.json with F1-score data and all data organized in different tables.
    
    Returns:
        Complete evaluation data including F1 metrics, confusion matrix, and detailed results organized in tables
    """
    try:
        # Reload data from JSON file
        current_data = load_evaluation_results()
        results = current_data.get('results', [])
        
        # Calculate F1 metrics for each toxic label
        label_metrics = {
            'toxic': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
            'severe_toxic': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
            'obscene': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
            'threat': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
            'insult': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
            'identity_hate': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
        }
        
        # Analyze each result for label-wise metrics
        for result in results:
            ground_truth_labels = result.get('toxic_labels', {})
            detected_labels = result.get('detected_labels', {})
            
            for label in label_metrics.keys():
                gt_has_label = ground_truth_labels.get(label, 0) == 1
                pred_has_label = label in detected_labels
                
                if gt_has_label and pred_has_label:
                    label_metrics[label]['tp'] += 1
                elif gt_has_label and not pred_has_label:
                    label_metrics[label]['fn'] += 1
                elif not gt_has_label and pred_has_label:
                    label_metrics[label]['fp'] += 1
                else:
                    label_metrics[label]['tn'] += 1
        
        # Calculate F1 metrics for each label
        f1_metrics_table = []
        for label, metrics in label_metrics.items():
            tp, fp, fn, tn = metrics['tp'], metrics['fp'], metrics['fn'], metrics['tn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            f1_metrics_table.append({
                'label': label,
                'precision': round(precision, 3),
                'recall': round(recall, 3),
                'f1_score': round(f1_score, 3),
                'accuracy': round(accuracy, 3),
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'true_negatives': tn,
                'support': tp + fn
            })
        
        # Sort by F1 score descending
        f1_metrics_table.sort(key=lambda x: x['f1_score'], reverse=True)
        
        # Calculate macro averages
        macro_precision = sum(item['precision'] for item in f1_metrics_table) / len(f1_metrics_table)
        macro_recall = sum(item['recall'] for item in f1_metrics_table) / len(f1_metrics_table)
        macro_f1 = sum(item['f1_score'] for item in f1_metrics_table) / len(f1_metrics_table)
        macro_accuracy = sum(item['accuracy'] for item in f1_metrics_table) / len(f1_metrics_table)
        
        # Overall confusion matrix
        overall_confusion_matrix = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0
        }
        
        for result in results:
            ground_truth = result.get('ground_truth', '')
            predicted = result.get('predicted', '')
            
            if ground_truth == 'offensive' and predicted == 'offensive':
                overall_confusion_matrix['true_positives'] += 1
            elif ground_truth == 'offensive' and predicted == 'non-offensive':
                overall_confusion_matrix['false_negatives'] += 1
            elif ground_truth == 'non-offensive' and predicted == 'offensive':
                overall_confusion_matrix['false_positives'] += 1
            elif ground_truth == 'non-offensive' and predicted == 'non-offensive':
                overall_confusion_matrix['true_negatives'] += 1
        
        # Calculate overall metrics
        tp = overall_confusion_matrix['true_positives']
        fp = overall_confusion_matrix['false_positives']
        tn = overall_confusion_matrix['true_negatives']
        fn = overall_confusion_matrix['false_negatives']
        
        overall_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        overall_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        overall_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # Create detailed results table
        detailed_results_table = []
        for i, result in enumerate(results, 1):
            detailed_results_table.append({
                'id': i,
                'text': result.get('text', '')[:100] + '...' if len(result.get('text', '')) > 100 else result.get('text', ''),
                'ground_truth': result.get('ground_truth', ''),
                'predicted': result.get('predicted', ''),
                'correct': result.get('correct', False),
                'detected_labels': result.get('detected_labels', {}),
                'toxic_labels': result.get('toxic_labels', {}),
                'analysis_type': result.get('llm_response', {}).get('analysis_type', ''),
                'confidence': result.get('llm_response', {}).get('confidence', '')
            })
        
        # Create summary statistics
        total_examples = len(results)
        correct_predictions = sum(1 for r in results if r.get('correct', False))
        incorrect_predictions = total_examples - correct_predictions
        
        # Label distribution analysis
        label_distribution = {}
        for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
            label_count = sum(1 for r in results if r.get('toxic_labels', {}).get(label, 0) == 1)
            label_distribution[label] = {
                'count': label_count,
                'percentage': round((label_count / total_examples) * 100, 2) if total_examples > 0 else 0
            }
        
        # Analysis type distribution
        analysis_type_distribution = {}
        for result in results:
            analysis_type = result.get('llm_response', {}).get('analysis_type', 'unknown')
            analysis_type_distribution[analysis_type] = analysis_type_distribution.get(analysis_type, 0) + 1
        
        # Confidence distribution
        confidence_distribution = {}
        for result in results:
            confidence = result.get('llm_response', {}).get('confidence', 'unknown')
            confidence_distribution[confidence] = confidence_distribution.get(confidence, 0) + 1
        
        comprehensive_results = {
            "message": "Comprehensive Evaluation Results from evaluation_results.json",
            "metadata": {
                "total_examples": total_examples,
                "confidence_threshold": current_data.get('confidence_threshold', 0.5),
                "data_source": "evaluation_results.json",
                "evaluation_date": "From JSON file",
                "model_version": "LLM Enhanced Toxic Classifier"
            },
            "overall_performance": {
                "accuracy": round(overall_accuracy, 3),
                "precision": round(overall_precision, 3),
                "recall": round(overall_recall, 3),
                "f1_score": round(overall_f1, 3),
                "correct_predictions": correct_predictions,
                "incorrect_predictions": incorrect_predictions,
                "success_rate": round((correct_predictions / total_examples) * 100, 2) if total_examples > 0 else 0
            },
            "f1_metrics_table": f1_metrics_table,
            "macro_averages": {
                "macro_precision": round(macro_precision, 3),
                "macro_recall": round(macro_recall, 3),
                "macro_f1": round(macro_f1, 3),
                "macro_accuracy": round(macro_accuracy, 3)
            },
            "confusion_matrix": overall_confusion_matrix,
            "label_distribution": label_distribution,
            "analysis_type_distribution": analysis_type_distribution,
            "confidence_distribution": confidence_distribution,
            "detailed_results_table": detailed_results_table,
            "summary_statistics": {
                "total_tests": total_examples,
                "passed_tests": correct_predictions,
                "failed_tests": incorrect_predictions,
                "success_rate_percentage": round((correct_predictions / total_examples) * 100, 2) if total_examples > 0 else 0,
                "best_performing_label": f1_metrics_table[0]['label'] if f1_metrics_table else 'N/A',
                "best_f1_score": f1_metrics_table[0]['f1_score'] if f1_metrics_table else 0.0,
                "worst_performing_label": f1_metrics_table[-1]['label'] if f1_metrics_table else 'N/A',
                "worst_f1_score": f1_metrics_table[-1]['f1_score'] if f1_metrics_table else 0.0
            },
            "tables_description": {
                "f1_metrics_table": "Detailed F1 metrics for each toxic label with precision, recall, accuracy, and confusion matrix values",
                "detailed_results_table": "Individual test results with text samples, predictions, and metadata",
                "confusion_matrix": "Overall binary classification confusion matrix (offensive vs non-offensive)",
                "label_distribution": "Distribution of toxic labels in the test dataset",
                "analysis_type_distribution": "Distribution of analysis methods used (rule-based vs LLM)",
                "confidence_distribution": "Distribution of confidence levels in predictions"
            },
            "endpoint_info": {
                "description": "Comprehensive evaluation results endpoint with F1-score data and all data organized in different tables",
                "usage": "GET /evaluation/results",
                "data_source": "evaluation_results.json",
                "available_tables": [
                    "f1_metrics_table",
                    "detailed_results_table", 
                    "confusion_matrix",
                    "label_distribution",
                    "analysis_type_distribution",
                    "confidence_distribution"
                ]
            }
        }
        
        return comprehensive_results
        
    except Exception as e:
        logger.error(f"Error in evaluation results: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation results failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 