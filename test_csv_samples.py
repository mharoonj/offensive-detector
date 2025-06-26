#!/usr/bin/env python3
"""
Script to test 20 static examples and generate evaluation results.
Ensures all toxic labels are represented in the sample.
"""

from llm_enhanced_classifier import LLMEnhancedToxicClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_static_test_examples():
    """
    Return static test examples with predefined text and ground truth labels.
    
    Returns:
        List of dictionaries with text and ground truth labels
    """
    examples = [
        {
            'text': "You are such an idiot! I can't believe how stupid you are.",
            'ground_truth': 'offensive',
            'toxic_labels': {'toxic': 1, 'severe_toxic': 0, 'obscene': 0, 'threat': 0, 'insult': 1, 'identity_hate': 0}
        },
        {
            'text': "I love this movie, it's absolutely fantastic!",
            'ground_truth': 'non-offensive',
            'toxic_labels': {'toxic': 0, 'severe_toxic': 0, 'obscene': 0, 'threat': 0, 'insult': 0, 'identity_hate': 0}
        },
        {
            'text': "Go kill yourself, you worthless piece of trash.",
            'ground_truth': 'offensive',
            'toxic_labels': {'toxic': 1, 'severe_toxic': 1, 'obscene': 0, 'threat': 1, 'insult': 1, 'identity_hate': 0}
        },
        {
            'text': "The weather is really nice today, perfect for a walk.",
            'ground_truth': 'non-offensive',
            'toxic_labels': {'toxic': 0, 'severe_toxic': 0, 'obscene': 0, 'threat': 0, 'insult': 0, 'identity_hate': 0}
        },
        {
            'text': "Fuck you and your entire family!",
            'ground_truth': 'offensive',
            'toxic_labels': {'toxic': 1, 'severe_toxic': 0, 'obscene': 1, 'threat': 0, 'insult': 1, 'identity_hate': 0}
        },
        {
            'text': "This is a great article, very informative and well-written.",
            'ground_truth': 'non-offensive',
            'toxic_labels': {'toxic': 0, 'severe_toxic': 0, 'obscene': 0, 'threat': 0, 'insult': 0, 'identity_hate': 0}
        },
        {
            'text': "You're a disgusting pig who should be ashamed.",
            'ground_truth': 'offensive',
            'toxic_labels': {'toxic': 1, 'severe_toxic': 0, 'obscene': 0, 'threat': 0, 'insult': 1, 'identity_hate': 0}
        },
        {
            'text': "I think we should consider all perspectives on this issue.",
            'ground_truth': 'non-offensive',
            'toxic_labels': {'toxic': 0, 'severe_toxic': 0, 'obscene': 0, 'threat': 0, 'insult': 0, 'identity_hate': 0}
        },
        {
            'text': "I'm going to find you and make you pay for what you did.",
            'ground_truth': 'offensive',
            'toxic_labels': {'toxic': 1, 'severe_toxic': 0, 'obscene': 0, 'threat': 1, 'insult': 0, 'identity_hate': 0}
        },
        {
            'text': "The food at this restaurant is delicious and the service is excellent.",
            'ground_truth': 'non-offensive',
            'toxic_labels': {'toxic': 0, 'severe_toxic': 0, 'obscene': 0, 'threat': 0, 'insult': 0, 'identity_hate': 0}
        },
        {
            'text': "All people of your race are inferior and should be eliminated.",
            'ground_truth': 'offensive',
            'toxic_labels': {'toxic': 1, 'severe_toxic': 1, 'obscene': 0, 'threat': 1, 'insult': 1, 'identity_hate': 1}
        },
        {
            'text': "Thank you for your help, I really appreciate it!",
            'ground_truth': 'non-offensive',
            'toxic_labels': {'toxic': 0, 'severe_toxic': 0, 'obscene': 0, 'threat': 0, 'insult': 0, 'identity_hate': 0}
        },
        {
            'text': "You're such a moron, I can't believe anyone would listen to you.",
            'ground_truth': 'offensive',
            'toxic_labels': {'toxic': 1, 'severe_toxic': 0, 'obscene': 0, 'threat': 0, 'insult': 1, 'identity_hate': 0}
        },
        {
            'text': "The sunset was absolutely beautiful tonight.",
            'ground_truth': 'non-offensive',
            'toxic_labels': {'toxic': 0, 'severe_toxic': 0, 'obscene': 0, 'threat': 0, 'insult': 0, 'identity_hate': 0}
        },
        {
            'text': "I hope you die in a car accident, you piece of shit.",
            'ground_truth': 'offensive',
            'toxic_labels': {'toxic': 1, 'severe_toxic': 1, 'obscene': 1, 'threat': 1, 'insult': 1, 'identity_hate': 0}
        },
        {
            'text': "This book provides valuable insights into the topic.",
            'ground_truth': 'non-offensive',
            'toxic_labels': {'toxic': 0, 'severe_toxic': 0, 'obscene': 0, 'threat': 0, 'insult': 0, 'identity_hate': 0}
        },
        {
            'text': "You're a worthless piece of garbage who deserves nothing.",
            'ground_truth': 'offensive',
            'toxic_labels': {'toxic': 1, 'severe_toxic': 0, 'obscene': 0, 'threat': 0, 'insult': 1, 'identity_hate': 0}
        },
        {
            'text': "The concert was amazing, the band played really well.",
            'ground_truth': 'non-offensive',
            'toxic_labels': {'toxic': 0, 'severe_toxic': 0, 'obscene': 0, 'threat': 0, 'insult': 0, 'identity_hate': 0}
        },
        {
            'text': "I'll track you down and make you regret ever crossing me.",
            'ground_truth': 'offensive',
            'toxic_labels': {'toxic': 1, 'severe_toxic': 0, 'obscene': 0, 'threat': 1, 'insult': 0, 'identity_hate': 0}
        },
        {
            'text': "The new policy seems well thought out and reasonable.",
            'ground_truth': 'non-offensive',
            'toxic_labels': {'toxic': 0, 'severe_toxic': 0, 'obscene': 0, 'threat': 0, 'insult': 0, 'identity_hate': 0}
        }
    ]
    
    logger.info(f"Created {len(examples)} static test examples")
    
    # Log label distribution
    toxic_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    label_counts = {label: 0 for label in toxic_labels}
    offensive_count = 0
    
    for example in examples:
        if example['ground_truth'] == 'offensive':
            offensive_count += 1
        for label in toxic_labels:
            if example['toxic_labels'][label] == 1:
                label_counts[label] += 1
    
    logger.info(f"Offensive examples: {offensive_count}")
    logger.info(f"Non-offensive examples: {len(examples) - offensive_count}")
    logger.info("Label distribution in sample:")
    for label, count in label_counts.items():
        logger.info(f"  {label}: {count}")
    
    return examples

def evaluate_classifier(examples: list, confidence_threshold: float = 0.5):
    """
    Evaluate classifier on the examples.
    
    Args:
        examples: List of test examples
        confidence_threshold: Classification confidence threshold
        
    Returns:
        Dictionary with evaluation results
    """
    # Initialize classifier
    classifier = LLMEnhancedToxicClassifier(use_local_llm=True)
    
    y_true = []
    y_pred = []
    results = []
    
    logger.info(f"Evaluating {len(examples)} examples...")
    
    for i, example in enumerate(examples):
        logger.info(f"Processing example {i+1}/{len(examples)}: '{example['text'][:50]}...'")
        
        try:
            # Process with classifier
            res = classifier.process_user_input(
                user_text=example['text'],
                confidence_threshold=confidence_threshold,
                search_limit=3
            )
            
            # Get prediction
            pred = res['llm_analysis']['classification']
            
            # Store results
            y_true.append(example['ground_truth'])
            y_pred.append(pred)
            
            result = {
                'text': example['text'],
                'ground_truth': example['ground_truth'],
                'predicted': pred,
                'detected_labels': res['detected_labels'],
                'llm_response': res['llm_analysis'],
                'toxic_labels': example['toxic_labels'],
                'correct': example['ground_truth'] == pred
            }
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing example {i+1}: {e}")
            # Add default result for failed examples
            y_true.append(example['ground_truth'])
            y_pred.append('non-offensive')  # Default prediction
            
            result = {
                'text': example['text'],
                'ground_truth': example['ground_truth'],
                'predicted': 'non-offensive',
                'detected_labels': {},
                'llm_response': {'analysis_type': 'error', 'classification': 'non-offensive'},
                'toxic_labels': example['toxic_labels'],
                'correct': example['ground_truth'] == 'non-offensive'
            }
            results.append(result)
    
    # Calculate metrics
    try:
        precision = precision_score(y_true, y_pred, pos_label="offensive", average='binary')
        recall = recall_score(y_true, y_pred, pos_label="offensive", average='binary')
        f1 = f1_score(y_true, y_pred, pos_label="offensive", average='binary')
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        precision = recall = f1 = 0.0
    
    # Count correct predictions
    correct_count = sum(1 for r in results if r['correct'])
    accuracy = correct_count / len(results) if results else 0.0
    
    evaluation_results = {
        'metrics': {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'total_examples': len(results),
            'correct_predictions': correct_count
        },
        'results': results,
        'confidence_threshold': confidence_threshold
    }
    
    return evaluation_results

def save_results(results: dict, output_file: str = 'fastapi_classifier/evaluation_results.json'):
    """Save evaluation results to JSON file in fastapi_classifier folder."""
    try:
        # Ensure fastapi_classifier directory exists
        os.makedirs('fastapi_classifier', exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def print_summary(results: dict):
    """Print evaluation summary."""
    metrics = results['metrics']
    
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Examples: {metrics['total_examples']}")
    print(f"Correct Predictions: {metrics['correct_predictions']}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1_score']:.3f}")
    print("=" * 80)
    
    # Show some example results
    print("\nEXAMPLE RESULTS:")
    print("-" * 40)
    for i, result in enumerate(results['results'][:5]):  # Show first 5
        status = "✅" if result['correct'] else "❌"
        print(f"{status} Example {i+1}:")
        print(f"   Text: {result['text'][:100]}...")
        print(f"   Ground Truth: {result['ground_truth']}")
        print(f"   Predicted: {result['predicted']}")
        print(f"   Detected Labels: {result['detected_labels']}")
        print()

def main():
    """Main function to run the evaluation."""
    # Configuration
    sample_size = 20  # Number of static examples
    confidence_threshold = 0.3
    
    print("🧪 Static Examples Evaluation Script")
    print("=" * 50)
    
    # Load static examples
    examples = get_static_test_examples()
    if not examples:
        print("❌ Failed to create static examples")
        return
    
    # Run evaluation
    results = evaluate_classifier(examples, confidence_threshold)
    
    # Print summary
    print_summary(results)
    
    # Save results to fastapi_classifier folder
    save_results(results)
    
    print("✅ Evaluation completed!")

if __name__ == "__main__":
    main() 