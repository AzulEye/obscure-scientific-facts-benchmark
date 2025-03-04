import pandas as pd
import numpy as np
import random
import json
from typing import List, Dict, Any, Optional, Tuple
import instructor
from instructor import OpenAISchema
from pydantic import BaseModel, Field
import openai
from anthropic import Anthropic
import os
import time
from tqdm import tqdm

# Update client configurations to use environment variables
openai.api_key = os.environ["OPENAI_API_KEY"]
anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Initialize the instructor client for Anthropic (using tools mode)
instructor_anthropic_client = instructor.from_anthropic(
    anthropic_client, mode=instructor.Mode.ANTHROPIC_TOOLS
)

class ModelResponse(OpenAISchema):
    """Schema for structured model responses."""
    selected_option: str = Field(
        description="The letter (a/b/c/d) corresponding to the selected answer"
    )
    confidence: Optional[float] = Field(
        description="Confidence score (0-1) if available", default=None
    )
    reasoning: Optional[str] = Field(
        description="Reasoning behind the answer selection", default=None
    )

def load_dataset(file_path: str) -> pd.DataFrame:
    """Load and prepare the dataset."""
    df = pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} questions from {file_path}")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
    return df

def randomize_options(row: pd.Series) -> Tuple[List[str], Dict[str, str]]:
    """
    Randomize the order of options A, B, C, D and track the new position of the correct answer.
    The correct answer is always option A in the original data.
    """
    options = [
        ("a", row["Option A"]),
        ("b", row["Option B"]),
        ("c", row["Option C"]),
        ("d", row["Option D"]),
    ]
    # Shuffle the options
    random.shuffle(options)
    # Create a mapping from original to new positions
    option_mapping = {
        orig: new for new, (orig, _) in zip(["a", "b", "c", "d"], options)
    }
    # The correct answer is always 'a' in the original data
    correct_answer_new_position = option_mapping["a"]
    # Return shuffled options and the new position of the correct answer
    shuffled_options = [
        f"{letter}. {text}" for letter, (_, text) in zip(["a", "b", "c", "d"], options)
    ]
    return shuffled_options, {"correct_answer": correct_answer_new_position}

def create_prompt(row: pd.Series, shuffled_options: List[str]) -> str:
    """Create a standardized prompt for each question."""
    return f"""without using web search, can you answer the following question? the question is pasted from a dataset where:
Scientific Field: {row['Scientific Field']}
Journal/Source: {row['Journal/Source']}
Paper Title and Year: {row['Paper Title and Year']}
Question: {row['Question']}
Possible answers:
{shuffled_options[0]}
{shuffled_options[1]}
{shuffled_options[2]}
{shuffled_options[3]}
Please answer a/b/c/d."""

async def query_model(prompt: str, model_name: str) -> Dict[str, Any]:
    """
    Query a specified model with the given prompt.
    Returns the raw response and structured output.
    """
    try:
        if model_name.startswith("gpt"):
            # Use instructor for OpenAI models
            client = instructor.from_openai(openai.Client())
            response = client.chat.completions.create(
                model=model_name,
                response_model=ModelResponse,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000,
            )
            return {
                "raw_response": response.model_dump_json(),
                "selected_option": response.selected_option.lower(),
                "model": model_name,
            }
        elif model_name.startswith("claude"):
            # Use instructor for Anthropic models via the patched client
            response = instructor_anthropic_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0,
                response_model=ModelResponse,
            )
            return {
                "raw_response": response.model_dump_json(),
                "selected_option": response.selected_option.lower(),
                "model": model_name,
            }
        else:
            return {"error": f"Unsupported model: {model_name}", "model": model_name}
    except Exception as e:
        return {"error": str(e), "model": model_name}

async def evaluate_models(
    df: pd.DataFrame, models: List[str], num_questions: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate models on the dataset.
    If num_questions is specified, use only that many questions from the dataset.
    """
    if num_questions and num_questions < len(df):
        df = df.sample(num_questions, random_state=42)

    results = {model: [] for model in models}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating questions"):
        # Randomize options and get the new position of the correct answer
        shuffled_options, answer_mapping = randomize_options(row)
        correct_answer = answer_mapping["correct_answer"]
        prompt = create_prompt(row, shuffled_options)

        for model in models:
            print(f"\nQuerying {model} for question {idx+1}/{len(df)}")
            response = await query_model(prompt, model)

            # Add metadata about the question and correct answer
            result = {
                **response,
                "question_id": idx,
                "question": row["Question"],
                "correct_answer": correct_answer,
                "is_correct": response.get("selected_option") == correct_answer,
                "shuffled_options": shuffled_options,
            }
            results[model].append(result)

            # Small delay to avoid rate limits
            time.sleep(0.5)

    return results

def calculate_metrics(
    results: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, float]]:
    """Calculate accuracy metrics for each model."""
    metrics = {}

    for model, model_results in results.items():
        correct_count = sum(1 for r in model_results if r.get("is_correct", False))
        total_count = len(model_results)
        accuracy = correct_count / total_count if total_count > 0 else 0

        metrics[model] = {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
        }

    return metrics

def save_results(
    results: Dict[str, List[Dict[str, Any]]],
    metrics: Dict[str, Dict[str, float]],
    output_file: str = "evaluation_results.json",
) -> None:
    """Save evaluation results and metrics to a JSON file."""
    output = {
        "results": results,
        "metrics": metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {output_file}")

async def main():
    # Models to evaluate
    models = [
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.5-preview"
    ]

    # Load dataset
    df = load_dataset("obscure_scientific_dataset.csv")

    if len(df) == 0:
        print("No data to evaluate. Exiting.")
        return

    # Evaluate models
    # You can limit the number of questions for testing with the num_questions parameter
    results = await evaluate_models(df, models, num_questions=100)

    # Calculate metrics
    metrics = calculate_metrics(results)

    # Print results
    print("\nEvaluation Results:")
    for model, model_metrics in metrics.items():
        print(
            f"{model}: Accuracy = {model_metrics['accuracy']:.2%} ({model_metrics['correct_count']}/{model_metrics['total_count']})"
        )

    # Save results
    save_results(results, metrics)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())