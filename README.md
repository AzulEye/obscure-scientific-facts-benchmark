# Obscure Scientific Facts Benchmark

This project evaluates Large Language Models (LLMs) on their ability to recall obscure scientific facts from research published in 2022. The benchmark provides insights into how model scale correlates with scientific knowledge retention.

## 📊 Key Result

On our dataset of 165 obscure scientific facts from 2022, models performed as follows:

| Model | Accuracy |
|-------|----------|
| gpt-4o-mini | 55% |
| gpt-4o | 69% |
| gpt-4.5-preview | 94% |

These results demonstrate a clear correlation between model scale and recall of obscure scientific facts, with gpt-4.5-preview showing remarkable capacity to retain detailed scientific information.

## Why This Matters

The scaling pattern suggests that larger, more sophisticated language models develop increasingly comprehensive grasp of scientific literature. This capability is particularly valuable for:

- AI systems trained to function as virtual scientific assistants
- Models that need to accurately recall specific experimental results and methodologies
- Systems designed to help researchers navigate expanding scientific knowledge bases
- Maybe some day we get to a point where models can innovate by connecting seemingly unrelated pieces of scientific information.

## 🛠 Requirements

- Python 3.7+
- Required packages:
  ```
  pandas
  numpy
  openai
  anthropic
  tqdm
  pydantic
  instructor
  ```

## 🚀 Getting Started

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/obscure-scientific-facts-benchmark.git
   cd obscure-scientific-facts-benchmark
   ```

2. Install required packages
   ```bash
   pip install -r requirements.txt
   ```

3. Set up API keys as environment variables
   ```bash
   export OPENAI_API_KEY='your_openai_api_key'
   export ANTHROPIC_API_KEY='your_anthropic_api_key'
   ```

### Running Evaluations

Run the evaluation script:
```bash
python run_evaluations.py
```

Results will be saved to `evaluation_results.json`.

## 📝 Dataset Format

The dataset (`obscure_scientific_dataset.csv`) contains multiple-choice questions about scientific findings from 2022 papers across various fields. Each entry includes:

- Scientific Field
- Journal/Source
- Paper Title and Year
- Question
- Four possible answers (Options A-D)
- Correct Answer (A is always the correct answer)

## 📋 Dataset Creation Process

The dataset was created through the following process:

1. **Fact Collection**: Used OpenAI's Deep Research feature to identify obscure scientific facts from top publications in 2022.

2. **Exact prompt to deep research**:
   ```
"I want to create a dataset of obscure scientific facts of papers published in 2022 in various fields like physics, math, biology, computer science, etc. The facts should be stuff that are pretty specific (ideally numerical, and if not numerical, then a multiple choice question) and based either on empirical experiments or logical deduction (in the case of math). The way I want to achieve this is to get some top 100 scientific publications, and extract as many facts as you can find."
   - Example facts:
     - Question: In the paper "Why do deep convolutional networks generalize so poorly to small image transformations?" what is the percentage of times the antialiased model changed its prediction? Answer: 15%
     - Question: In the paper "Why do deep convolutional networks generalize so poorly to small image transformations?" how did the authors measure image typicality?
       a. perceptual similarity of an image to the 10 nearest neighbors in the training set.
       b. L2 distance in pixel space to the 10 nearest neighbors in the training set.
       c. L2 distance in pixel space to the 5 nearest neighbors in the training set.
       d. perceptual similarity of an image to the 5 nearest neighbors in the training set.
   ```

3. **Multiple-Choice Formatting**: Facts were transformed into multiple-choice questions with specific, often numerical answers.

## 🧩 Core Components

- **load_dataset()**: Loads questions from the CSV file
- **randomize_options()**: Shuffles answer options to prevent positional bias
- **create_prompt()**: Formats questions for model querying
- **query_model()**: Handles API interactions with different LLM providers
- **evaluate_models()**: Processes model responses and calculates performance
- **save_results()**: Exports evaluation data for analysis

## 🔄 Extending the Benchmark

To extend this benchmark:
- Add more obscure facts from recent papers
- Include different model providers
- Expand to facts from different years to test temporal knowledge

## 📄 License

This project is licensed under the MIT License

## 👥 Contributing

Contributions are welcome!