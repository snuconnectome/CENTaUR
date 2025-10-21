#!/usr/bin/env python3
"""
Ko-CENTaUR Evaluation Metrics

Implements core evaluation metrics:
- NLL (Negative Log-Likelihood)
- Token-level accuracy
- Korean norm correlation
- Age effect validation
- Clinical discrimination (AUC)
"""

import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM

class CENTaURMetrics:
    """Evaluation metrics for Ko-CENTaUR"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def compute_nll_accuracy(self, test_data: List[Dict]) -> Dict:
        """
        Compute NLL and accuracy on masked tokens only

        Args:
            test_data: List of test samples with 'text' and 'participant' fields

        Returns:
            dict: {'nll': float, 'accuracy': float, 'n_tokens': int}
        """
        total_nll = 0.0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for sample in test_data:
                # Extract masked regions << >>
                text = sample['text']
                masked_spans = self._extract_masked_spans(text)

                # Compute NLL and accuracy for each span
                for span in masked_spans:
                    nll, correct, n_tokens = self._compute_span_metrics(text, span)
                    total_nll += nll
                    total_correct += correct
                    total_tokens += n_tokens

        return {
            'nll': total_nll / total_tokens if total_tokens > 0 else float('inf'),
            'accuracy': total_correct / total_tokens if total_tokens > 0 else 0.0,
            'n_tokens': total_tokens
        }

    def evaluate_korean_norms(self, test_data: List[Dict], korean_norms: Dict) -> Dict:
        """
        Validate against Korean normative data

        Args:
            test_data: Test samples with age and scores
            korean_norms: Korean normative scores by age group

        Returns:
            dict: {'correlation': float, 'p_value': float}
        """
        predicted_scores = []
        actual_scores = []

        for sample in test_data:
            # Get model prediction
            pred_score = self._predict_score(sample)
            predicted_scores.append(pred_score)

            # Get actual score
            actual = sample['questionnaire_metadata']['total_score']
            actual_scores.append(actual)

        # Pearson correlation
        r, p = pearsonr(predicted_scores, actual_scores)

        return {
            'correlation': r,
            'p_value': p,
            'n_samples': len(test_data)
        }

    def evaluate_age_effects(self, test_data: List[Dict]) -> Dict:
        """
        Validate age-related performance patterns

        Returns:
            dict: {'age_correlation': float, 'p_value': float}
        """
        ages = []
        scores = []

        for sample in test_data:
            age_months = sample['participant']['age_months']
            score = self._predict_score(sample)

            ages.append(age_months)
            scores.append(score)

        r, p = pearsonr(ages, scores)

        return {
            'age_correlation': r,
            'p_value': p,
            'direction': 'positive' if r > 0 else 'negative'
        }

    def evaluate_clinical_discrimination(self, test_data: List[Dict]) -> Dict:
        """
        Evaluate clinical vs control discrimination

        Returns:
            dict: {'auc': float, 'accuracy': float}
        """
        clinical_labels = []
        predicted_probs = []

        for sample in test_data:
            # Clinical label (1 = clinical, 0 = control)
            is_clinical = sample['participant'].get('clinical_group') != 'control'
            clinical_labels.append(int(is_clinical))

            # Model prediction (higher score = more severe)
            prob = self._predict_clinical_probability(sample)
            predicted_probs.append(prob)

        # AUC
        auc = roc_auc_score(clinical_labels, predicted_probs)

        # Accuracy at threshold 0.5
        predictions = [1 if p > 0.5 else 0 for p in predicted_probs]
        acc = accuracy_score(clinical_labels, predictions)

        return {
            'auc': auc,
            'accuracy': acc,
            'n_clinical': sum(clinical_labels),
            'n_control': len(clinical_labels) - sum(clinical_labels)
        }

    def full_evaluation_report(self, test_data: List[Dict], korean_norms: Dict = None) -> Dict:
        """Complete evaluation suite"""
        report = {
            'nll_accuracy': self.compute_nll_accuracy(test_data),
            'age_effects': self.evaluate_age_effects(test_data),
        }

        if korean_norms:
            report['korean_norms'] = self.evaluate_korean_norms(test_data, korean_norms)

        # Clinical discrimination if labels available
        if any('clinical_group' in s['participant'] for s in test_data):
            report['clinical_discrimination'] = self.evaluate_clinical_discrimination(test_data)

        return report

    # Helper methods (to be implemented)
    def _extract_masked_spans(self, text: str) -> List[str]:
        """Extract text within << >> markers"""
        import re
        return re.findall(r'<<(.+?)>>', text)

    def _compute_span_metrics(self, text: str, span: str) -> tuple:
        """Compute NLL and accuracy for a masked span"""
        # Placeholder implementation
        return 0.0, 0, len(span.split())

    def _predict_score(self, sample: Dict) -> float:
        """Predict total score from sample"""
        # Placeholder implementation
        return 0.0

    def _predict_clinical_probability(self, sample: Dict) -> float:
        """Predict clinical probability"""
        # Placeholder implementation
        return 0.5
