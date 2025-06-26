from transformers import pipeline

def detect_categories(text: str, confidence_threshold: float = 0.5):
    """
    Stage 1: Use lightweight zero-shot classifier to detect potential categories.
    
    Args:
        text: Input text to classify
        confidence_threshold: Minimum confidence score to consider a category
        
    Returns:
        Dictionary of detected categories with their confidence scores
    """
    # Load zero-shot classifier
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    # Define candidate labels for different types of content
    candidate_labels = [
        "insult", "sexism", "threat", "harassment", "racism", 
        "homophobia", "bullying", "spam", "non-offensive"
    ]
    
    # Classify input text
    result = classifier(text, candidate_labels)
    
    # Filter categories based on confidence threshold
    detected_categories = {}
    for label, score in zip(result["labels"], result["scores"]):
        if score > confidence_threshold:
            detected_categories[label] = score
    
    return detected_categories

def main():
    # Test cases
    test_texts = [
        "You are a bitch and I hope you die",
        "This is a great product, buy now for 50% off!",
        "Women should stay in the kitchen",
        "Hello, how are you today?",
        "I'm going to hurt you if you don't do what I say"
    ]
    
    print("=== Category Detection Results ===\n")
    
    for i, text in enumerate(test_texts, 1):
        print(f"Test Case {i}: '{text}'")
        print("-" * 50)
        
        # Detect categories
        detected_categories = detect_categories(text)
        
        if detected_categories:
            print("Detected Categories:")
            for category, confidence in detected_categories.items():
                print(f"  - {category}: {confidence:.3f}")
        else:
            print("No categories detected above threshold")
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()