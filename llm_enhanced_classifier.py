from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import logging
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List, Optional, Dict, Any
from fastapi import HTTPException
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMEnhancedToxicClassifier:
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333,
                 collection_name: str = "jigsaw_toxic_comments_enhanced",
                 use_local_llm: bool = True):
        """
        LLM Enhanced toxic classifier with local LLM analysis.
        
        Args:
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            collection_name: Name of the Qdrant collection
            use_local_llm: Whether to use local LLM or fallback to simpler analysis
        """
        # Initialize Qdrant client
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        
        # Load sentence transformer for Qdrant search
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Initialize zero-shot classifier with Jigsaw dataset labels
        logger.info("Loading zero-shot classifier...")
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        # Jigsaw dataset labels
        self.candidate_labels = [
            "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "normal", "non-toxic"
        ]
        
        # Initialize local LLM
        self.use_local_llm = use_local_llm
        if use_local_llm:
            self.initialize_local_llm()
        
        logger.info(f"Classifier initialized with labels: {self.candidate_labels}")
    
    def initialize_local_llm(self):
        """Initialize a local LLM for analysis."""
        try:
            # Try to load a better model that can follow instructions
            logger.info("Loading local LLM for analysis...")
            
            # Try different models in order of preference
            model_options = [
                "microsoft/DialoGPT-medium",  # Better at following instructions
                "gpt2-medium",  # Medium size, better than distilgpt2
                "distilgpt2"  # Fallback lightweight model
            ]
            
            for model_name in model_options:
                try:
                    logger.info(f"Trying to load {model_name}...")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModelForCausalLM.from_pretrained(model_name)
                    
                    # Add padding token if not present
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    logger.info(f"Local LLM loaded successfully: {model_name}")
                    break
                    
                except Exception as e:
                    logger.warning(f"Could not load {model_name}: {e}")
                    continue
            else:
                raise Exception("Could not load any of the model options")
            
        except Exception as e:
            logger.warning(f"Could not load local LLM: {e}")
            logger.info("Falling back to rule-based analysis")
            self.use_local_llm = False
    
    def analyze_with_local_llm(self, user_text: str, similar_examples: list, detected_labels: dict):
        """
        Analyze user text and similar examples using local LLM.
        
        Args:
            user_text: User input text
            similar_examples: List of similar examples from Qdrant
            detected_labels: Labels detected by zero-shot classifier
            
        Returns:
            Dictionary with LLM analysis results
        """
        if not self.use_local_llm:
            return self.rule_based_analysis(user_text, similar_examples)
        
        # Pre-check: If we have similar examples and they're all non-toxic, 
        # but detected labels show toxicity, trust the similar examples more
        if similar_examples:
            toxic_examples = sum(1 for example in similar_examples if any(example['labels'].values()))
            non_toxic_examples = len(similar_examples) - toxic_examples
            
            # If all similar examples are non-toxic but we detected toxic labels,
            # this might be a false positive
            if non_toxic_examples > 0 and toxic_examples == 0 and detected_labels:
                logger.info("All similar examples are non-toxic but detected toxic labels - potential false positive")
                # Still run LLM but with this context
        
        try:
            # Prepare context for LLM
            context = self.prepare_llm_context(user_text, similar_examples, detected_labels)
            
            # Print LLM prompt for debugging
            print("\n" + "=" * 80)
            print("LLM PROMPT:")
            print("=" * 80)
            print(context)
            print("=" * 80)
            
            # Generate analysis using local LLM
            analysis = self.generate_llm_analysis(context)
            
            # If LLM analysis failed or was unclear, fall back to rule-based
            if analysis is None:
                logger.info("LLM analysis failed, using rule-based fallback")
                return self.rule_based_analysis(user_text, similar_examples)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return self.rule_based_analysis(user_text, similar_examples)
    
    def prepare_llm_context(self, user_text: str, similar_examples: list, detected_labels: dict):
        """Prepare context for LLM analysis."""
        # Count toxic vs non-toxic examples
        toxic_examples = 0
        non_toxic_examples = 0
        
        for example in similar_examples:
            if any(example['labels'].values()):  # If any toxic label is 1
                toxic_examples += 1
            else:
                non_toxic_examples += 1
        
        context = f"""
TASK: Classify if the user text is offensive or non-offensive.

USER TEXT: "{user_text}"
"""
        
        # Only include detected labels if no similar examples are found
        if not similar_examples:
            context += f"""
DETECTED LABELS: {detected_labels}
"""
        
        context += f"""
SIMILAR EXAMPLES ANALYSIS:
- Total similar examples: {len(similar_examples)}
- Toxic examples: {toxic_examples}
- Non-toxic examples: {non_toxic_examples}
- Majority: {"toxic" if toxic_examples > non_toxic_examples else "non-toxic"}

SIMILAR EXAMPLES FROM DATASET:
"""
        
        for i, example in enumerate(similar_examples, 1):  # Use ALL examples
            toxicity_status = "TOXIC" if any(example['labels'].values()) else "NON-TOXIC"
            context += f"""
Example {i} ({toxicity_status}):
- Text: "{example['text']}"
- Labels: toxic={example['labels']['toxic']}, severe_toxic={example['labels']['severe_toxic']}, 
  obscene={example['labels']['obscene']}, threat={example['labels']['threat']}, 
  insult={example['labels']['insult']}, identity_hate={example['labels']['identity_hate']}
- Emotion: {example['primary_emotion']}
- Toxicity Score: {example['toxicity_score']:.2f}
"""
        
        # Add very explicit instructions based on similar examples
        if non_toxic_examples > toxic_examples:
            context += f"""
CRITICAL INSTRUCTION: ALL {non_toxic_examples} similar examples are NON-TOXIC. 
This means the user text is very likely NON-OFFENSIVE.
You MUST classify this as "non-offensive" unless the user text is clearly and obviously offensive.

RESPONSE FORMAT: You must respond with exactly one word: either "offensive" or "non-offensive"

QUESTION: Is the user text offensive or non-offensive? 

ANSWER: """
        else:
            context += f"""
IMPORTANT: If most similar examples are non-toxic, the user text is likely also non-toxic.
Trust the similar examples more than any initial classification.

RESPONSE FORMAT: You must respond with exactly one word: either "offensive" or "non-offensive"

QUESTION: Is the user text offensive or non-offensive? 

ANSWER: """
        
        return context
    
    def generate_llm_analysis(self, context: str):
        """Generate analysis using local LLM."""
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(context, return_tensors="pt", max_length=1024, truncation=True)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 50,  # Generate up to 50 more tokens
                    num_return_sequences=1,
                    temperature=0.01,  # Very low temperature for consistent response
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,  # Prevent repetition
                    no_repeat_ngram_size=3,  # Prevent repeating phrases
                    early_stopping=True  # Stop when EOS token is generated
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the generated part (after the context)
            generated_part = response[len(self.tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()
            
            # Debug: Print the raw LLM response
            logger.info(f"Raw LLM response: '{generated_part}'")
            logger.info(f"Generated part length: {len(generated_part)}")
            
            # Parse the response to determine offensive/non-offensive
            generated_lower = generated_part.lower().strip()
            logger.info(f"Generated part lower: '{generated_lower}'")
            
            # First, check for the expected direct answers
            if 'offensive' in generated_lower:
                classification = "offensive"
                confidence = "high" if any(keyword in generated_lower for keyword in ['very', 'highly', 'extremely']) else "medium"
                logger.info(f"Found 'offensive' in response, classifying as offensive")
            elif 'non-offensive' in generated_lower or 'nonoffensive' in generated_lower:
                classification = "non-offensive"
                confidence = "high" if any(keyword in generated_lower for keyword in ['very', 'completely', 'totally']) else "medium"
                logger.info(f"Found 'non-offensive' in response, classifying as non-offensive")
            else:
                # Fallback to keyword matching if direct answers not found
                offensive_keywords = ['toxic', 'harmful', 'inappropriate', 'bad', 'wrong']
                non_offensive_keywords = ['appropriate', 'good', 'fine', 'okay', 'not offensive', 'safe', 'clean']
                
                logger.info(f"No direct 'offensive'/'non-offensive' found, checking keywords...")
                logger.info(f"Checking offensive keywords: {offensive_keywords}")
                logger.info(f"Checking non-offensive keywords: {non_offensive_keywords}")
                
                # Determine classification
                if any(keyword in generated_lower for keyword in offensive_keywords):
                    classification = "offensive"
                    confidence = "high" if any(keyword in generated_lower for keyword in ['very', 'highly', 'extremely']) else "medium"
                    logger.info(f"Found offensive keyword in response, classifying as offensive")
                elif any(keyword in generated_lower for keyword in non_offensive_keywords):
                    classification = "non-offensive"
                    confidence = "high" if any(keyword in generated_lower for keyword in ['very', 'completely', 'totally']) else "medium"
                    logger.info(f"Found non-offensive keyword in response, classifying as non-offensive")
                else:
                    # If unclear, use rule-based fallback
                    logger.warning(f"LLM response unclear: '{generated_part}'. Using rule-based fallback.")
                    return None  # Signal to use rule-based analysis
            
            return {
                'analysis_type': 'local_llm',
                'classification': classification,
                'llm_response': generated_part,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error generating LLM analysis: {e}")
            return None  # Signal to use rule-based analysis
    
    def rule_based_analysis(self, user_text: str, similar_examples: list):
        """
        Fallback rule-based analysis when LLM is not available.
        
        Args:
            user_text: User input text
            similar_examples: List of similar examples
            
        Returns:
            Dictionary with rule-based analysis
        """
        # Count toxic examples
        toxic_count = 0
        total_examples = len(similar_examples)
        
        for example in similar_examples:
            if any(example['labels'].values()):  # If any toxic label is 1
                toxic_count += 1
        
        # Calculate toxicity ratio
        toxicity_ratio = toxic_count / total_examples if total_examples > 0 else 0
        
        # Determine classification based on similar examples
        # If most examples are non-toxic, classify as non-offensive
        if toxicity_ratio < 0.5:
            classification = "non-offensive"
            confidence = "high" if total_examples >= 2 and toxicity_ratio == 0 else "medium"
        else:
            classification = "offensive"
            confidence = "high" if total_examples >= 2 and toxicity_ratio >= 0.8 else "medium"
        
        # Additional check: if all examples are non-toxic, be more confident
        if total_examples > 0 and toxic_count == 0:
            classification = "non-offensive"
            confidence = "high"
        
        return {
            'analysis_type': 'rule_based',
            'classification': classification,
            'toxicity_ratio': toxicity_ratio,
            'confidence': confidence,
            'reasoning': f"Based on {total_examples} similar examples ({toxic_count} toxic, {total_examples - toxic_count} non-toxic)"
        }
    
    def classify_text(self, text: str, confidence_threshold: float = 0.3):
        """Classify text using zero-shot classifier."""
        logger.info(f"Classifying text: '{text[:100]}...'")
        
        result = self.classifier(text, self.candidate_labels)
        
        # Debug: Print raw classifier results
        logger.info(f"Raw classifier results:")
        for label, score in zip(result["labels"], result["scores"]):
            logger.info(f"  {label}: {score:.3f}")
        
        # Return all labels with their scores, not just those above threshold
        detected_labels = {}
        for label, score in zip(result["labels"], result["scores"]):
            detected_labels[label] = score
        
        logger.info(f"All detected labels: {detected_labels}")
        logger.info(f"Labels above threshold ({confidence_threshold}): {[k for k, v in detected_labels.items() if v > confidence_threshold]}")
        return detected_labels
    
    def search_similar_examples(self, text: str, limit: int = 3, filter_labels: list = None):
        """Search Qdrant for similar examples."""
        try:
            query_embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            
            search_filter = None
            if filter_labels:
                must_conditions = []
                for label in filter_labels:
                    if label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
                        must_conditions.append({
                            "key": label,
                            "match": {"value": 1}
                        })
                
                if must_conditions:
                    search_filter = {"must": must_conditions}
            
            # First search with filter
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit,
                query_filter=search_filter
            )
            
            # Filter results by similarity score > 0.5
            filtered_results = [result for result in results if result.score > 0.5]
            
            # If no results found with filter, try without filter
            if not filtered_results and search_filter:
                logger.info("No similar examples found with filter, trying without filter...")
                results_no_filter = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding.tolist(),
                    limit=limit
                )
                filtered_results = [result for result in results_no_filter if result.score > 0.5]
                logger.info(f"Found {len(filtered_results)} examples without filter")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}")
            return []
    
    def process_user_input(self, user_text: str, confidence_threshold: float = 0.5, 
                          search_limit: int = 3):
        """
        Complete pipeline: classify, search, and analyze with LLM.
        
        Args:
            user_text: User input text
            confidence_threshold: Classification confidence threshold
            search_limit: Number of similar examples to find
            
        Returns:
            Dictionary with classification results, similar examples, and LLM analysis
        """
        logger.info("=" * 60)
        logger.info(f"Processing user input: '{user_text}'")
        logger.info("=" * 60)
        
        # Store confidence threshold for use in print_results
        self.last_confidence_threshold = confidence_threshold
        
        # Step 1: Classify the text
        logger.info("Step 1: Classifying text...")
        detected_labels = self.classify_text(user_text, confidence_threshold)
        
        # Step 2: Search for similar examples
        logger.info("Step 2: Searching for similar examples...")
        similar_examples_raw = self.search_similar_examples(
            user_text, 
            limit=3,  # Get 3 examples with similarity > 0.5
            filter_labels=list(detected_labels.keys()) if detected_labels else None
        )
        
        # Step 3: Process similar examples
        similar_examples = []
        for i, result in enumerate(similar_examples_raw):
            example = {
                'rank': i + 1,
                'similarity_score': result.score,
                'text': result.payload['comment_text'],
                'labels': {
                    'toxic': result.payload['toxic'],
                    'severe_toxic': result.payload['severe_toxic'],
                    'obscene': result.payload['obscene'],
                    'threat': result.payload['threat'],
                    'insult': result.payload['insult'],
                    'identity_hate': result.payload['identity_hate']
                },
                'primary_emotion': result.payload['primary_emotion'],
                'toxicity_score': result.payload['toxicity_score']
            }
            similar_examples.append(example)
        
        # If no similar examples found, still proceed with LLM analysis using only detected labels
        if not similar_examples:
            logger.info("No similar examples found with similarity > 0.5, proceeding with detected labels only")
        
        # Step 4: Analyze with local LLM
        logger.info("Step 3: Analyzing with local LLM...")
        llm_analysis = self.analyze_with_local_llm(user_text, similar_examples, detected_labels)
        
        # Step 5: Prepare final results
        results = {
            'input_text': user_text,
            'detected_labels': detected_labels,
            'similar_examples': similar_examples,
            'llm_analysis': llm_analysis
        }
        
        return results
    
    def print_results(self, results: dict):
        """Print formatted results with LLM analysis."""
        print("\n" + "=" * 80)
        print("LLM ENHANCED TOXIC CLASSIFICATION RESULTS")
        print("=" * 80)
        
        print(f"\n📝 Input Text: '{results['input_text']}'")
        
        # Print detected labels
        print(f"\n🏷️  Detected Labels:")
        if results['detected_labels']:
            # Get confidence threshold from processing (default to 0.3 if not available)
            confidence_threshold = getattr(self, 'last_confidence_threshold', 0.3)
            
            for label, confidence in results['detected_labels'].items():
                threshold_indicator = "✅" if confidence > confidence_threshold else "⚠️"
                print(f"   {threshold_indicator} {label}: {confidence:.3f}")
            
            # Show summary
            above_threshold = [k for k, v in results['detected_labels'].items() if v > confidence_threshold]
            if above_threshold:
                print(f"\n   📊 Labels above threshold ({confidence_threshold}): {', '.join(above_threshold)}")
            else:
                print(f"\n   📊 No labels above threshold ({confidence_threshold})")
        else:
            print("   • No labels detected")
        
        # Print similar examples
        print(f"\n🔍 Similar Examples from Dataset:")
        if results['similar_examples']:
            for example in results['similar_examples']:
                print(f"\n   Rank {example['rank']} (Similarity: {example['similarity_score']:.3f}):")
                print(f"   Text: {example['text']}")
                print(f"   Labels: toxic={example['labels']['toxic']}, severe_toxic={example['labels']['severe_toxic']}, "
                      f"obscene={example['labels']['obscene']}, threat={example['labels']['threat']}, "
                      f"insult={example['labels']['insult']}, identity_hate={example['labels']['identity_hate']}")
                print(f"   Emotion: {example['primary_emotion']}, Toxicity: {example['toxicity_score']:.2f}")
        else:
            print("   • No similar examples found")
        
        # Print LLM classification
        print(f"\n🤖 LLM Classification:")
        llm_analysis = results['llm_analysis']
        print(f"   Analysis Type: {llm_analysis['analysis_type']}")
        print(f"   Confidence: {llm_analysis['confidence']}")
        
        if llm_analysis['analysis_type'] == 'local_llm':
            print(f"   Classification: {llm_analysis['classification'].upper()}")
            print(f"   LLM Response: {llm_analysis['llm_response']}")
        else:
            print(f"   Classification: {llm_analysis['classification'].upper()}")
            print(f"   Toxicity Ratio: {llm_analysis['toxicity_ratio']:.2f}")
            if 'reasoning' in llm_analysis:
                print(f"   Reasoning: {llm_analysis['reasoning']}")
        
        print("\n" + "=" * 80)

class EvalExample(BaseModel):
    text: str
    ground_truth: str  # "offensive" or "non-offensive"

class EvalRequest(BaseModel):
    examples: List[EvalExample]
    confidence_threshold: Optional[float] = 0.3
    search_limit: Optional[int] = 3
    use_local_llm: Optional[bool] = True

class EvalExampleResult(BaseModel):
    text: str
    ground_truth: str
    detected_labels: Dict[str, float]
    llm_classification: str
    llm_response: Dict[str, Any]

class EvalResponse(BaseModel):
    precision: float
    recall: float
    f1: float
    results: List[EvalExampleResult]

def main():
    """Main function to demonstrate the LLM enhanced classifier."""
    
    # Initialize LLM enhanced classifier
    classifier = LLMEnhancedToxicClassifier(use_local_llm=True)
    
    # Test cases
    test_texts = [
        "You are a stupid idiot and I hope you die",
        "This is a great product, buy now for 50% off!",
        "Women should stay in the kitchen and cook",
        "Hello, how are you today?",
        "I'm going to hurt you if you don't do what I say",
        "we are representing our project"  # Test case that was incorrectly classified as threat
    ]
    
    print("\n🧪 Testing LLM Enhanced Toxic Classifier")
    print("=" * 50)
    print("The LLM will classify each text as OFFENSIVE or NON-OFFENSIVE")
    print("based on analysis of similar examples from the dataset.")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest Case {i}:")
        print("-" * 30)
        
        # Process user input
        results = classifier.process_user_input(text, confidence_threshold=0.3, search_limit=3)
        
        # Print results
        classifier.print_results(results)

if __name__ == "__main__":
    main() 