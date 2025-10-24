#!/usr/bin/env python3
import torch
import json

# Load results from different experiments
results_dirs = [
    '/scratch/connectome/connectome1/ko-centaur/results/choices13k_100_fixed',
    '/scratch/connectome/connectome1/ko-centaur/results/choices13k_test',
    '/scratch/connectome/connectome1/ko-centaur/results/full_eval_test_fixed'
]

all_data = {}

for result_dir in results_dirs:
    exp_name = result_dir.split("/")[-1]
    print(f'\n{"="*80}')
    print(f'Results from: {exp_name}')
    print("="*80)

    try:
        all_results = torch.load(f'{result_dir}/all_results.pth', weights_only=False)
        all_data[exp_name] = all_results

        for model_name, metrics in all_results.items():
            print(f'\n{model_name}:')
            if isinstance(metrics.get("mean_accuracy"), float):
                print(f'  Mean Accuracy: {metrics["mean_accuracy"]:.4f}')
            if isinstance(metrics.get("std_accuracy"), float):
                print(f'  Std Accuracy: {metrics["std_accuracy"]:.4f}')
            if 'fold_results' in metrics and len(metrics['fold_results']) > 0:
                print(f'  Number of folds: {len(metrics["fold_results"])}')
                # Calculate average log likelihood
                log_liks = [f.get('test_log_likelihood', 0) for f in metrics['fold_results'] if isinstance(f.get('test_log_likelihood'), (int, float))]
                if log_liks:
                    print(f'  Mean Log Likelihood: {sum(log_liks)/len(log_liks):.4f}')
    except Exception as e:
        print(f'  Error loading results: {e}')

# Save aggregated data
print('\n' + '='*80)
print('Saving aggregated data...')
torch.save(all_data, '/tmp/aggregated_results.pth')
print('Saved to /tmp/aggregated_results.pth')
