import os
import argparse
import json
from utils import set_seed, Timer
from train import train_and_evaluate, evaluate_on_test
from config import NUM_FOLDS, RESULTS_DIR, MODEL_SAVE_PATH


def main():
    parser = argparse.ArgumentParser(description='Bone Fracture Classification')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'train_test'],
                        help='Mode to run: train, test, or both')
    parser.add_argument('--folds', type=int, default=NUM_FOLDS,
                        help='Number of cross-validation folds')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Directory containing model checkpoints for testing')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed()
    
    # Create timer to track execution time
    timer = Timer()
    
    # Train mode
    if args.mode in ['train', 'train_test']:
        print(f"Starting training with {args.folds} folds...")
        timer.start()
        
        # Train and evaluate models
        results = train_and_evaluate(folds=args.folds)
        
        # Save results
        results_file = os.path.join(RESULTS_DIR, 'cv_results.json')
        with open(results_file, 'w') as f:
            # Convert numpy values to Python native types for JSON serialization
            json_results = {
                'overall_metrics': {
                    k: float(v) for k, v in results['overall_metrics'].items()
                },
                'fold_metrics': [
                    {k: float(v) if k != 'confusion_matrix' else v.tolist() 
                     for k, v in fold_metrics.items()}
                    for fold_metrics in results['fold_metrics']
                ],
                'fold_models': results['fold_models']
            }
            json.dump(json_results, f, indent=4)
        
        timer.stop()
        print(f"Training completed in {timer.get_elapsed_time():.2f} seconds")
        print(f"Results saved to {results_file}")
        
        # Store model paths for testing
        model_paths = results['fold_models']
    
    # Test mode
    if args.mode in ['test', 'train_test']:
        # If only testing, use provided model directory
        if args.mode == 'test':
            if args.model_dir is None:
                raise ValueError("Please provide a model directory for testing")
            
            # Get model paths from the provided directory
            model_paths = [
                os.path.join(args.model_dir, f)
                for f in os.listdir(args.model_dir)
                if f.endswith('.pth')
            ]
            
            if not model_paths:
                raise ValueError(f"No model checkpoint found in {args.model_dir}")
        
        print(f"Starting testing with {len(model_paths)} models...")
        timer.start()
        
        # Evaluate on test data
        test_metrics = evaluate_on_test(model_paths)
        
        # Save test results
        test_results_file = os.path.join(RESULTS_DIR, 'test_results.json')
        with open(test_results_file, 'w') as f:
            json.dump(
                {k: float(v) if k != 'confusion_matrix' else v.tolist() 
                 for k, v in test_metrics.items()},
                f, indent=4
            )
        
        timer.stop()
        print(f"Testing completed in {timer.get_elapsed_time():.2f} seconds")
        print(f"Test results saved to {test_results_file}")


if __name__ == "__main__":
    main()
