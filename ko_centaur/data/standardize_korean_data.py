#!/usr/bin/env python3
"""
Standardize heterogeneous lab data to Ko-CENTaUR JSONL format

Converts Excel/CSV/SPSS files from collaborating labs into:
- Standardized JSONL format
- EXAONE instruction template
- Masked response format (<< >>)
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List

def convert_kmmse_to_jsonl(df: pd.DataFrame, lab_id: str) -> List[Dict]:
    """Convert K-MMSE data to JSONL format"""
    records = []

    for idx, row in df.iterrows():
        # Build instruction text
        text_parts = ["[meta: "]
        text_parts.append(f"age_m={row['age_months']}; ")
        text_parts.append(f"sex={row['gender']}]")
        text_parts.append("\n\n지남력 평가:\n")

        # Add Q&A pairs
        if not pd.isna(row.get('year')):
            text_parts.append(f"오늘은 몇 년도인가요? <<{row['year']}>>\n")
        if not pd.isna(row.get('month')):
            text_parts.append(f"오늘은 몇 월인가요? <<{row['month']}>>\n")

        record = {
            "text": "".join(text_parts),
            "experiment": "k_mmse",
            "participant": {
                "id": f"{lab_id}_{idx:04d}",
                "age_months": int(row['age_months']),
                "age_group": get_age_group(row['age_months']),
                "gender": row['gender']
            },
            "questionnaire_metadata": {
                "instrument": "K-MMSE",
                "total_score": row.get('total_score', None),
                "normative_percentile": row.get('percentile', None)
            },
            "source_lab": lab_id
        }

        records.append(record)

    return records

def get_age_group(age_months: int) -> str:
    """Classify age into developmental groups"""
    if age_months < 72:
        return "child"
    elif age_months < 144:
        return "child"
    elif age_months < 216:
        return "adolescent"
    elif age_months < 312:
        return "young_adult"
    else:
        return "adult"

def validate_record(record: Dict) -> tuple[bool, str]:
    """Validate a single record for completeness"""
    required_fields = ['text', 'experiment', 'participant', 'source_lab']

    for field in required_fields:
        if field not in record:
            return False, f"Missing field: {field}"

    # Check masked responses
    if '<<' not in record['text'] or '>>' not in record['text']:
        return False, "No masked responses found"

    return True, "OK"

def main():
    """Example standardization workflow"""
    print("Korean Data Standardization Pipeline")
    print("="*60)

    # Example: Convert K-MMSE data
    # input_file = "data/raw/LAB1_kmmse.xlsx"
    # df = pd.read_excel(input_file)
    # records = convert_kmmse_to_jsonl(df, "LAB1")

    # Save to JSONL
    # output_file = "data/processed/LAB1_kmmse_standardized.jsonl"
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     for record in records:
    #         f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print("✅ Template functions ready")
    print("Add lab-specific conversion functions as needed")

if __name__ == "__main__":
    main()
