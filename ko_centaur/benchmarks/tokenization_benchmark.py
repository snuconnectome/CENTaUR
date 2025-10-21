#!/usr/bin/env python3
"""
Tokenization Efficiency Benchmark for Korean Language Models

Compares tokenization efficiency of:
- EXAONE-3.0-7.8B-Instruct
- EEVE-Korean-10.8B-v1.0
- Llama-3.1-8B-Instruct
- Qwen2.5-7B-Instruct

Metric: Tokens per character for Korean text
Lower is better (more efficient)
"""

from transformers import AutoTokenizer
import json
from pathlib import Path

# Korean test corpus from various psychological assessments
TEST_CORPUS = {
    "k_mmse": """
    지남력 평가:
    오늘은 몇 년도인가요? 2025년입니다.
    지금은 몇 월인가요? 1월입니다.
    오늘은 며칠인가요? 7일입니다.
    오늘은 무슨 요일인가요? 화요일입니다.
    지금 계절은 무엇입니까? 겨울입니다.

    기억력 평가:
    제가 세 가지 물건 이름을 말씀드리겠습니다. 잘 기억해 주십시오.
    비행기, 연필, 소나무. 다시 한번 말씀드리겠습니다.
    비행기, 연필, 소나무. 이제 따라 말씀해 보십시오.
    """,

    "phq9": """
    지난 2주 동안 다음의 문제들로 인해서 얼마나 자주 방해를 받았는지 표시해 주십시오:

    1. 일 또는 여가 활동을 하는데 흥미나 즐거움을 느끼지 못함
    2. 기분이 가라앉거나, 우울하거나, 희망이 없음
    3. 잠이 들거나 계속 잠을 자는 것이 어려움, 또는 잠을 너무 많이 잠
    4. 피곤하다고 느끼거나 기운이 거의 없음
    5. 식욕이 줄었거나 과식을 함
    """,

    "sdq": """
    자녀의 행동에 관한 질문입니다. 지난 6개월간 자녀의 행동을 가장 잘 설명하는 답에 표시해 주십시오:

    1. 다른 사람들의 감정을 배려한다
    2. 침착하지 못하고 과잉 행동을 보이거나 오래 앉아 있지 못한다
    3. 자주 두통, 복통 또는 구역질을 호소한다
    4. 다른 아이들과 잘 나누어 가진다
    5. 화가 나면 불같이 성을 내고 소리를 지른다
    """,

    "experiment_instructions": """
    당신은 이제부터 여러 가지 기하학적 도형들을 보게 될 것입니다.
    당신의 과제는 각 도형이 E 범주에 속하는지 K 범주에 속하는지를 구분하는 규칙을 학습하는 것입니다.
    각 도형이 제시될 때마다, 해당 도형이 어느 범주에 속한다고 생각하는지 키를 눌러 응답해 주세요.
    그러면 정답 피드백을 받게 됩니다.
    가능한 한 많은 정답을 맞히도록 노력해 주세요.
    """
}

def benchmark_tokenizer(model_name: str, tokenizer_kwargs: dict = None) -> dict:
    """Benchmark a single tokenizer"""
    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}

    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

        results = {
            "model": model_name,
            "vocab_size": len(tokenizer),
            "tasks": {}
        }

        total_chars = 0
        total_tokens = 0

        for task_name, text in TEST_CORPUS.items():
            # Remove extra whitespace
            text = " ".join(text.split())

            # Tokenize
            tokens = tokenizer.encode(text)

            # Calculate metrics
            num_chars = len(text)
            num_tokens = len(tokens)
            efficiency = num_tokens / num_chars

            results["tasks"][task_name] = {
                "characters": num_chars,
                "tokens": num_tokens,
                "tokens_per_char": round(efficiency, 4)
            }

            total_chars += num_chars
            total_tokens += num_tokens

            print(f"\n{task_name}:")
            print(f"  Characters: {num_chars}")
            print(f"  Tokens: {num_tokens}")
            print(f"  Tokens/char: {efficiency:.4f}")

        # Overall metrics
        overall_efficiency = total_tokens / total_chars
        results["overall"] = {
            "total_characters": total_chars,
            "total_tokens": total_tokens,
            "tokens_per_char": round(overall_efficiency, 4)
        }

        print(f"\n{'Overall':}")
        print(f"  Total characters: {total_chars}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Tokens/char: {overall_efficiency:.4f}")

        return results

    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")
        return {"model": model_name, "error": str(e)}

def main():
    print("="*60)
    print("Korean Tokenization Efficiency Benchmark")
    print("="*60)
    print("\nLower tokens/char = more efficient")

    # Models to test
    models = [
        {
            "name": "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
            "kwargs": {"trust_remote_code": True}
        },
        {
            "name": "yanolja/EEVE-Korean-10.8B-v1.0",
            "kwargs": {}
        },
        {
            "name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "kwargs": {}
        },
        {
            "name": "Qwen/Qwen2.5-7B-Instruct",
            "kwargs": {}
        }
    ]

    all_results = []

    for model_config in models:
        result = benchmark_tokenizer(model_config["name"], model_config.get("kwargs"))
        all_results.append(result)

    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)

    # Sort by efficiency (lower is better)
    valid_results = [r for r in all_results if "overall" in r]
    valid_results.sort(key=lambda x: x["overall"]["tokens_per_char"])

    if valid_results:
        baseline = valid_results[-1]["overall"]["tokens_per_char"]  # Worst (Llama expected)

        print(f"\n{'Model':<40} {'Tokens/Char':<15} {'Efficiency vs Llama'}")
        print("-"*70)

        for result in valid_results:
            model_name = result["model"].split("/")[-1]
            tpc = result["overall"]["tokens_per_char"]
            relative = baseline / tpc

            print(f"{model_name:<40} {tpc:<15.4f} {relative:.2f}x")

    # Save results
    output_dir = Path(__file__).parent.parent / "docs" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "tokenization_benchmark_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to: {output_file}")

if __name__ == "__main__":
    main()
