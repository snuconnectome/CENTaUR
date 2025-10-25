#!/usr/bin/env python3
"""
Horizon Task Evaluation for Ko-CENTaUR
Evaluates exploration-exploitation trade-off using multi-armed bandit task
"""
import argparse
import time
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import json

num2words = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}

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
    elif model_name == "exaone":
        base_model = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
        adapter_path = "/scratch/connectome/connectome1/ko-centaur/outputs/exaone-qlora"
        local_files_only = False
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

def predict_choice(tokenizer, model, prompt: str):
    """Predict choice and extract hidden states"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Extract last layer hidden state from last token position
        hidden_states = outputs.hidden_states[-1]  # Last layer
        last_token_hidden = hidden_states[0, -1, :]  # Last token
        
        # Get logits for next token
        logits = outputs.logits[0, -1, :]
        
        # Get probabilities for choice tokens (1 vs 2)
        token_1 = tokenizer.encode("1", add_special_tokens=False)[0]
        token_2 = tokenizer.encode("2", add_special_tokens=False)[0]
        
        probs = torch.softmax(logits, dim=-1)
        prob_1 = probs[token_1].item()
        prob_2 = probs[token_2].item()
        
        # Normalize
        total = prob_1 + prob_2
        prob_1 = prob_1 / total
        prob_2 = prob_2 / total
    
    return prob_1, prob_2, last_token_hidden.cpu()

def evaluate_horizon_task(model_name: str, dataset: str, max_participants: int = 10):
    """Evaluate model on Horizon Task"""
    print(f"\n{'='*80}")
    print(f"Evaluating {model_name} on Horizon Task ({dataset})")
    print(f"{'='*80}\n")

    tokenizer, model = load_model(model_name)

    # Load data
    data_path = f"/scratch/connectome/connectome1/ko-centaur/data/downstream_tasks/horizon/{dataset}.csv"
    df = pd.read_csv(data_path)

    print(f"Loaded {len(df)} trials from {data_path}")

    num_participants = df.participant.max() + 1
    num_tasks = df.task.max() + 1

    print(f"Total participants: {num_participants}, Tasks per participant: {num_tasks}")

    # Sample participants if needed
    if max_participants and num_participants > max_participants:
        np.random.seed(42)
        sampled_participants = np.random.choice(num_participants, max_participants, replace=False)
        sampled_participants = sorted(sampled_participants)
        print(f"Sampling {max_participants} participants: {sampled_participants}")
    else:
        sampled_participants = list(range(num_participants))
        print(f"Using all {num_participants} participants")

    instructions = "You made the following observations in the past:\n"
    question = "Q: Which machine do you choose?\nA: Machine"

    all_results = []

    for participant in tqdm(sampled_participants, desc="Participants"):
        df_participant = df[df['participant'] == participant]
        
        for task in range(num_tasks):
            df_task = df_participant[df_participant['task'] == task]
            history = ""
            num_trials = df_task.trial.max() + 1
            
            for trial in range(num_trials):
                df_trial = df_task[df_task['trial'] == trial]
                
                if not df_trial['forced_choice'].item():
                    # Free choice trial
                    trials_left = num_trials - trial
                    if trials_left > 1:
                        trials_left_str = num2words[trials_left] + " additional choices"
                    else:
                        trials_left_str = num2words[trials_left] + " additional choice"
                    
                    trials_left_string = f"Your goal is to maximize the sum of received dollars within {trials_left_str}.\n\n"
                    prompt = instructions + history + "\n" + trials_left_string + question
                    
                    # Predict
                    prob_1, prob_2, hidden = predict_choice(tokenizer, model, prompt)
                    
                    # Ground truth
                    human_choice = int(df_trial.choice.item())
                    reward = df_trial.reward.item()
                    
                    # Store result
                    all_results.append({
                        'participant': participant,
                        'task': task,
                        'trial': trial,
                        'human_choice': human_choice,
                        'prob_machine_1': prob_1,
                        'prob_machine_2': prob_2,
                        'predicted_choice': 0 if prob_1 > prob_2 else 1,
                        'reward': reward,
                        'forced_choice': False,
                        'horizon': df_trial.horizon.item()
                    })
                
                # Update history
                c = int(df_trial.choice.item())
                r = df_trial.reward.item()
                
                if c == 0:
                    history += f"- Machine 1 delivered {r} dollars.\n"
                elif c == 1:
                    history += f"- Machine 2 delivered {r} dollars.\n"
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Calculate metrics
    accuracy = (results_df.human_choice == results_df.predicted_choice).mean()
    
    # Calculate negative log-likelihood
    nll_values = []
    for _, row in results_df.iterrows():
        if row.human_choice == 0:
            prob = row.prob_machine_1
        else:
            prob = row.prob_machine_2
        nll_values.append(-np.log(max(prob, 1e-10)))
    
    nll = np.mean(nll_values)
    
    print(f"\n{'='*80}")
    print(f"Results for {model_name} on {dataset}")
    print(f"{'='*80}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Negative Log-Likelihood: {nll:.4f}")
    print(f"Total free-choice trials evaluated: {len(results_df)}")
    
    # Save results
    output_dir = f"/scratch/connectome/connectome1/ko-centaur/results/horizon_{dataset}"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    results_df.to_csv(f"{output_dir}/{model_name}_predictions.csv", index=False)
    
    metrics = {
        'model': model_name,
        'dataset': dataset,
        'accuracy': float(accuracy),
        'nll': float(nll),
        'num_trials': len(results_df)
    }
    
    with open(f"{output_dir}/{model_name}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/")
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen25", "deepseek", "exaone"])
    parser.add_argument("--dataset", type=str, default="exp1", choices=["exp1", "exp2"])
    parser.add_argument("--max-participants", type=int, default=10,
                       help="Maximum number of participants to evaluate (default: 10 out of 31)")
    args = parser.parse_args()

    evaluate_horizon_task(args.model, args.dataset, args.max_participants)
