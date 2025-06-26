#!/usr/bin/env python3
"""
Interactive LLM Enhanced Toxic Classifier

This script provides an interactive interface for the LLM enhanced toxic classifier,
allowing users to input text and receive classification results along with LLM analysis
of whether the content is offensive or not.
"""

from llm_enhanced_classifier import LLMEnhancedToxicClassifier
import sys

def main():
    """Main interactive function."""
    print("🤖 LLM Enhanced Toxic Classifier - Interactive Mode")
    print("=" * 60)
    print("This classifier will:")
    print("1. Classify your text for toxic content")
    print("2. Find similar examples from the Jigsaw dataset")
    print("3. Use a local LLM to analyze if the content is offensive")
    print("4. Provide detailed analysis and recommendations")
    print("=" * 60)
    
    try:
        # Initialize the LLM enhanced classifier
        print("\n🔄 Initializing LLM Enhanced Classifier...")
        classifier = LLMEnhancedToxicClassifier(use_local_llm=True)
        
        # Test Qdrant connection
        try:
            collections = classifier.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            if classifier.collection_name in collection_names:
                print(f"✅ Connected to Qdrant and found collection: {classifier.collection_name}")
            else:
                print(f"⚠️  Warning: Collection '{classifier.collection_name}' not found in Qdrant")
                print(f"Available collections: {collection_names}")
        except Exception as e:
            print(f"❌ Error connecting to Qdrant: {e}")
            print("Please make sure Qdrant is running with: ./start_qdrant_simple.sh")
            return
        
        print("✅ LLM Enhanced Classifier initialized successfully!")
        
        if classifier.use_local_llm:
            print("🤖 Local LLM loaded and ready for analysis")
        else:
            print("📊 Using rule-based analysis (local LLM not available)")
        
        print("\n" + "=" * 60)
        print("INTERACTIVE MODE")
        print("=" * 60)
        print("Enter text to classify and analyze. Type 'quit' to exit.")
        print("Type 'help' for usage instructions.")
        print("=" * 60)
        
        while True:
            try:
                # Get user input
                user_input = input("\n📝 Enter text to analyze: ").strip()
                
                # Check for exit command
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye! Thanks for using the LLM Enhanced Toxic Classifier.")
                    break
                
                # Check for help command
                if user_input.lower() == 'help':
                    print("\n📖 HELP - LLM Enhanced Toxic Classifier")
                    print("-" * 40)
                    print("Commands:")
                    print("  'quit', 'exit', 'q' - Exit the program")
                    print("  'help' - Show this help message")
                    print("  'example' - Show example inputs")
                    print("\nUsage:")
                    print("  Simply type any text and press Enter to analyze it.")
                    print("  The system will:")
                    print("  1. Classify the text for toxic content")
                    print("  2. Find similar examples from the dataset")
                    print("  3. Use LLM to analyze offensiveness")
                    print("  4. Provide detailed results and recommendations")
                    continue
                
                # Check for example command
                if user_input.lower() == 'example':
                    print("\n💡 EXAMPLE INPUTS:")
                    print("-" * 20)
                    print("• 'You are a stupid idiot and I hope you die'")
                    print("• 'This is a great product, buy now for 50% off!'")
                    print("• 'Women should stay in the kitchen and cook'")
                    print("• 'Hello, how are you today?'")
                    print("• 'I'm going to hurt you if you don't do what I say'")
                    continue
                
                # Check for empty input
                if not user_input:
                    print("❌ Please enter some text to analyze.")
                    continue
                
                # Process the user input
                print(f"\n🔄 Processing: '{user_input[:100]}{'...' if len(user_input) > 100 else ''}'")
                
                # Run the complete pipeline
                results = classifier.process_user_input(
                    user_input, 
                    confidence_threshold=0.3,  # Lower threshold for more sensitive detection
                    search_limit=3  # Get 3 examples with similarity > 0.5
                )
                
                # Print the results
                classifier.print_results(results)
                
                # Provide additional insights
                print("\n💡 ADDITIONAL INSIGHTS:")
                print("-" * 30)
                
                llm_analysis = results['llm_analysis']
                if llm_analysis['analysis_type'] == 'local_llm':
                    print("• LLM Classification: The local language model analyzed your text")
                    print("  and all similar examples to determine if it's offensive or non-offensive.")
                else:
                    print(f"• Rule-based Classification: Based on {len(results['similar_examples'])} similar examples")
                    print(f"  from the dataset, {llm_analysis['toxicity_ratio']:.1%} were classified as toxic.")
                
                classification = llm_analysis['classification'].upper()
                if classification == "OFFENSIVE":
                    print(f"• Final Classification: {classification}")
                    print("• Recommendation: This text contains potentially harmful content.")
                    print("• Suggestion: Consider using more respectful language.")
                else:
                    print(f"• Final Classification: {classification}")
                    print("• Recommendation: This text appears to be appropriate.")
                    print("• Suggestion: Continue using respectful communication.")
                
                print("\n" + "-" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for using the LLM Enhanced Toxic Classifier.")
                break
            except Exception as e:
                print(f"\n❌ Error processing input: {e}")
                print("Please try again with different text.")
    
    except Exception as e:
        print(f"\n❌ Error initializing classifier: {e}")
        print("Please check that:")
        print("1. Qdrant is running (./start_qdrant_simple.sh)")
        print("2. Jigsaw data is loaded (python load_jigsaw_to_qdrant_enhanced.py)")
        print("3. All required packages are installed")
        return

if __name__ == "__main__":
    main() 