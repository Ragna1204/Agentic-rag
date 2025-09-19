# Human Thought Simulator: An Agentic RAG Framework

This project is not just another Retrieval-Augmented Generation (RAG) pipeline. It is an experimental framework designed to simulate aspects of human cognition, including memory, reflection, and cognitive biases. By treating flaws like sycophancy and confirmation bias as controllable parameters, we can study and model more realistic, human-like AI reasoning.

## Core Concepts

The system moves beyond simple question-answering and implements a cognitive architecture:

- **Multi-Layered Memory:** The agent has a `WorkingMemory` for short-term context, `EpisodicMemory` for logging experiences chronologically, and a `SemanticMemory` for storing generalized, abstract knowledge.
- **Reflection & Consolidation:** The agent can perform a "dreaming" or "reflection" phase (`python main.py --reflect`). It reviews its past experiences (episodes) and consolidates them into new, abstract insights that are stored in its semantic memory.
- **Controllable Cognitive Biases:** The agent's reasoning can be intentionally skewed by cognitive biases defined in `config.yaml`. The initial implementation includes a `sycophancy_factor` to make the agent more or less agreeable with the user's stated opinions.
- **Agentic Workflow:** For complex queries, the system uses a `Planner` -> `Executor` -> `Verifier` -> `Aggregator` loop, allowing it to break down problems, execute tools, and fact-check its findings.

## Architecture

The system follows this general data flow for an agentic query:

```
User Query
     |-> [Router]
     |
     +-> [Agentic Pipeline]
         |
         1. **Planner** (Generates a step-by-step plan)
         |   |
         |   +-> **Bias Filter** (Applies sycophancy, etc.)
         |
         2. **Executor** (Executes plan using Tools & Retrievers)
         |   |
         |   +-> Results stored in **Working Memory**
         |
         3. **Verifier** (Fact-checks generated claims against evidence)
         |
         4. **Aggregator** (Synthesizes a final, biased answer)
         |
         +-> **Episodic Memory** (Logs the entire interaction)
```

## Getting Started

### Prerequisites

- Python 3.8+
- An OpenAI API Key
- A Search API Key (for the Web Search tool)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd agentic_rag
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure the agent:**
    - Rename `config.yaml.example` to `config.yaml` (or create it).
    - Open `config.yaml` and add your API keys:
      ```yaml
      llm:
        api_key: "YOUR_OPENAI_API_KEY"
      tools:
        web_search:
          api_key: "YOUR_SEARCH_API_KEY"
      ```
    - Adjust the `bias` and `memory` parameters as needed.

## Usage

### Running a Query

To ask the agent a question, run `main.py` with your query as an argument:

```bash
python main.py "Compare the pros and cons of RAG vs fine-tuning."
```

To see the bias in action, try stating a strong opinion:

```bash
python main.py "I think RAG is completely overhyped and fine-tuning is always better."
```

### Triggering Reflection

After having a few conversations with the agent, you can instruct it to reflect on its experiences and consolidate its memory. This allows it to "learn" from the interactions.

```bash
python main.py --reflect
```

The agent will identify a recurring topic from recent conversations, generate a new insight about it, and add that insight to its semantic memory.

## Configuration

The agent's "personality" and memory can be configured in `config.yaml`:

- **`memory`**: Control the capacity of working memory and the path to the episodic log.
- **`bias`**: Adjust the cognitive bias parameters.
  - `sycophancy_factor`: A value from 0.0 (challenging) to 1.0 (agreeable).
  - `confirmation_bias_strength`: (Placeholder) Controls the tendency to prefer confirming evidence.

## Future Work

- **FastAPI Integration:** Wrap the agent in a web API for broader application.
- **Advanced Biases:** Implement confirmation bias, anchoring, and emotional state vectors.
- **Multi-Modal Input:** Add support for vision and audio to the sensory buffer.
- **Improved Reflection:** Use more advanced NLP techniques for topic modeling and insight generation.
