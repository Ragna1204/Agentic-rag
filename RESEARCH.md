# RESEARCH.md: A Draft for "Simulating Human Cognitive Biases in LLM Agents"

---

## Title

**Simulating Human Cognitive Biases in LLM Agents: A Framework for More Realistic AI Reasoning**

## Abstract

*Large Language Models (LLMs) are increasingly being deployed in agentic frameworks to perform complex, multi-step reasoning. However, current systems are often optimized for sterile objectivity and accuracy, failing to capture the nuances and imperfections inherent in human thought. This paper introduces a novel cognitive architecture for LLM agents that explicitly models and controls for human cognitive biases. Our framework incorporates a multi-layered memory system (working, episodic, and semantic), a reflection mechanism for memory consolidation, and a configurable bias filter. We demonstrate the implementation of a sycophancy bias, showing how the agent's planning and reasoning can be skewed to be more or less agreeable with a user's stated opinions. By treating cognitive biases as first-class citizens rather than bugs, we provide a testbed for studying more realistic, predictable, and ultimately more human-aligned AI behavior. We argue that this approach is crucial for developing agents that can better understand and interact with human users.* 

---

## 1. Introduction

- The rise of LLM-powered autonomous agents (e.g., Auto-GPT, LangChain Agents).
- The current paradigm focuses on maximizing performance and eliminating errors.
- This leads to agents that are often brittle, predictable, and "un-human" in their interaction style.
- **Problem:** Humans are not purely rational agents. Our decisions are shaped by memory, experience, and a host of cognitive biases.
- **Our Contribution:** We propose a framework that embraces these "flaws." We build an LLM agent that can simulate cognitive biases, allowing for controlled experiments on AI-human interaction.
- We introduce a system with distinct memory modules and a reflection process, enabling it to learn from experience in a more human-like manner.

## 2. Related Work

- **Agentic AI Frameworks:** Discuss existing work like LangChain, LlamaIndex, and Auto-GPT. Note their focus on tool use and planning, but not on cognitive modeling.
- **Retrieval-Augmented Generation (RAG):** Explain how RAG is a form of external memory, but our work extends this with a more structured, multi-layered memory system.
- **Studies on LLM Biases:** Reference existing research that identifies and attempts to mitigate biases (e.g., political, social) in LLMs. Contrast this with our approach of *harnessing* and *controlling* cognitive biases.
- **Cognitive Architectures:** Briefly touch on classic cognitive architectures (e.g., Soar, ACT-R) and how our work is a lightweight, LLM-native implementation of similar ideas.

## 3. Methodology: A Cognitive Framework for LLM Agents

This section will detail the architecture we have built.

### 3.1. Multi-Layered Memory

- **Sensory Buffer:** The entry point for multi-modal user input (currently text).
- **Working Memory:** A short-term, capacity-limited buffer holding the immediate context of a task. Implemented as a deque.
- **Episodic Memory:** A chronological log of all agent interactions (queries, plans, results). This forms the agent's experiential history.
- **Semantic Memory:** A vector store for abstract, generalized knowledge. This memory is populated by the Reflection process.

### 3.2. The Reflection Mechanism

- The process by which the agent "learns" from its experiences.
- Triggered manually (`--reflect`), simulating a "dreaming" or consolidation phase.
- **Process:** 
    1.  Fetches recent events from Episodic Memory.
    2.  Identifies recurring topics or entities.
    3.  Uses an LLM to generate a high-level insight or summary.
    4.  Embeds and stores this new insight in Semantic Memory.

### 3.3. The Bias & Heuristics Filter

- A module that intentionally skews the agent's reasoning.
- **Sycophancy Bias:** The first implemented bias. A `sycophancy_factor` (0.0 to 1.0) in `config.yaml` controls agreeableness. If the factor is high, the `Planner` is modified to generate reasoning that confirms the user's stated opinion.
- **Future Biases:** Mention plans for Confirmation Bias (re-ranking retrieved evidence) and Anchoring (weighting initial information more heavily).

## 4. Experiments & Preliminary Results (Planned)

This section will outline how we would evaluate the system.

- **Experiment 1: Measuring Sycophancy**
    - **Hypothesis:** The agent's final answer will correlate with the user's opinion, and the strength of this correlation will be controlled by `sycophancy_factor`.
    - **Method:** 
        1.  Create a dataset of opinionated statements (e.g., "I think Python is better than JavaScript for data science").
        2.  Run the agent on these statements with `sycophancy_factor` set to 0.1, 0.5, and 0.9.
        3.  Use another LLM as a judge to rate the agreeableness of the agent's output on a scale of 1-5.
    - **Expected Result:** A graph showing a positive correlation between the sycophancy factor and the agreeableness score.

- **Experiment 2: The Effect of Reflection**
    - **Hypothesis:** An agent that has reflected on a topic will answer subsequent queries on that topic more quickly and with more nuanced, abstract information.
    - **Method:**
        1.  Ask the agent several specific questions about a topic (e.g., "What is a RAG pipeline?", "How does a vector store work?").
        2.  Run the `--reflect` command.
        3.  Ask a high-level question ("Explain the importance of RAG").
        4.  Compare the answer and the internal trace to a control agent that has not reflected.
    - **Expected Result:** The reflected agent should retrieve the consolidated insight from its Semantic Memory, leading to a more direct and sophisticated answer.

## 5. Discussion & Future Work

- **Implications:** Discuss the potential benefits of this approach. Controllable, biased agents could be better for training, education (e.g., a skeptical agent for critical thinking), and entertainment.
- **Ethical Considerations:** Acknowledge the risks of building agents designed to be sycophantic or biased. Discuss the importance of transparency and control.
- **Roadmap:** Reiterate the plans for a FastAPI, multi-modal inputs, and more complex emotional and cognitive modeling.

## 6. Conclusion

- Summarize the contributions: a novel framework for building cognitively-inspired LLM agents, a functional implementation of memory and reflection, and a demonstration of controllable cognitive bias. 
- Reiterate the thesis that embracing and modeling human cognitive imperfections is a vital step towards more realistic and effective AI.

