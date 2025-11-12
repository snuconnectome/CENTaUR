#!/usr/bin/env python3
"""
Psych-101 Tasks Evaluation for Ko-CENTaUR
Evaluates models on selected tasks from Psych-101 dataset
"""
import argparse
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
from collections import defaultdict

# Task mappings based on CENTaUR paper analysis
TASK_EXPERIMENTS = {
    'twostep': ['ludwig2023human/exp0.csv'],
    'nback': ['cox2017information/exp1.csv'],
    'iowa_gambling': ['wood2005older/exp1.csv', 'horstmann2012iowa/exp1.csv'],
    'intertemporal': ['white2014decomposing/exp1.csv', 'lempert2019modeling/exp1.csv'],
    'decisions_description': ['erev2010choice/exp1.csv'],
    'decisions_experience': ['lejarraga2012learning/exp1.csv']
}

def load_model(model_name: str):
    """Load trained QLoRA model"""
    if model_name == "qwen25":
        base_model = "Qwen/Qwen2.5-32B-Instruct"
        adapter_path = "/scratch/connectome/connectome1/ko-centaur/outputs/qwen25-32b-qlora"
        local_files_only = False
    elif model_name == "deepseek":
        base_model = "/home/connectome/connectome1/models/deepseek-r1-distill-qwen-32b"
        adapter_path = "/scratch/connectome/connectome1/ko-centaur/outputs/deepseek-r1-qwen32b-qlora"
        local_files_only = True
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=local_files_only
    )

    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.eval()

    return tokenizer, model

def predict_next_token(tokenizer, model, prompt: str):
    """Predict next token probabilities"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=30000).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
    
    return probs.cpu(), logits.cpu()

def evaluate_task(tokenizer, model, task_name: str, data_path: str, max_samples: int = 100):
    """Evaluate model on a specific task"""
    print(f"\nEvaluating task: {task_name}")

    experiments = TASK_EXPERIMENTS.get(task_name, [])
    if not experiments:
        print(f"Warning: No experiments defined for {task_name}")
        return None

    # Load and filter data with early stopping
    all_samples = []
    with open(data_path, 'r') as f:
        for line in tqdm(f, desc=f"Loading {task_name}"):
            if len(all_samples) >= max_samples:
                break
            try:
                sample = json.loads(line)
                if sample['experiment'] in experiments:
                    all_samples.append(sample)
            except:
                continue

    if not all_samples:
        print(f"No samples found for {task_name}")
        return None

    print(f"Found {len(all_samples)} samples for {task_name}")

    # Evaluate samples
    results = []
    nll_values = []

    for sample in tqdm(all_samples, desc=f"Evaluating {task_name}"):
        text = sample['text']
        
        # Extract last choice from text (look for <<X>> pattern)
        import re
        choices = re.findall(r'<<([^>]+)>>', text)
        if not choices:
            continue
        
        last_choice = choices[-1]
        
        # Remove last choice from prompt
        prompt = text.rsplit(f'<<{last_choice}>>', 1)[0]
        
        # Predict
        probs, logits = predict_next_token(tokenizer, model, prompt)
        
        # Get probability for human choice
        choice_tokens = tokenizer.encode(last_choice, add_special_tokens=False)
        if choice_tokens:
            choice_token = choice_tokens[0]
            choice_prob = probs[choice_token].item()
            nll = -np.log(max(choice_prob, 1e-10))
            nll_values.append(nll)
            
            results.append({
                'experiment': sample['experiment'],
                'participant': sample['participant'],
                'human_choice': last_choice,
                'choice_prob': choice_prob,
                'nll': nll
            })
    
    if not nll_values:
        print(f"No valid predictions for {task_name}")
        return None
    
    avg_nll = np.mean(nll_values)
    print(f"{task_name} - Avg NLL: {avg_nll:.4f} ({len(results)} samples)")
    
    return {
        'task': task_name,
        'num_samples': len(results),
        'avg_nll': float(avg_nll),
        'results': results
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen25", "deepseek"])
    parser.add_argument("--tasks", type=str, nargs='+',
                       default=['twostep', 'nback', 'iowa_gambling', 'intertemporal'],
                       help="Tasks to evaluate")
    parser.add_argument("--data-path", type=str,
                       default="/scratch/connectome/connectome1/ko-centaur/data/psych101_train.jsonl")
    parser.add_argument("--max-samples", type=int, default=100,
                       help="Maximum samples per task (default: 100)")
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"Psych-101 Multi-Task Evaluation: {args.model}")
    print(f"Tasks: {', '.join(args.tasks)}")
    print(f"Max samples per task: {args.max_samples}")
    print(f"{'='*80}\n")

    tokenizer, model = load_model(args.model)

    all_results = {}
    for task in args.tasks:
        result = evaluate_task(tokenizer, model, task, args.data_path, args.max_samples)
        if result:
            all_results[task] = result
    
    # Save results
    import os
    output_dir = f"/scratch/connectome/connectome1/ko-centaur/results/psych101"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/{args.model}_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Summary for {args.model}")
    print(f"{'='*80}")
    for task, result in all_results.items():
        print(f"{task:30s} NLL: {result['avg_nll']:6.4f}  ({result['num_samples']} samples)")
    print(f"\nResults saved to {output_dir}/{args.model}_results.json")

if __name__ == "__main__":
    main()
