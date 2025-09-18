"""
Main entrypoint for the Advanced Agentic RAG Pipeline.

This script provides a command-line interface (CLI) to interact with the RAG system.
It orchestrates the different modules of the pipeline:
- Router: Decides whether to use a simple RAG or the full agentic workflow.
- Planner, Executor, Verifier, Aggregator: The core components of the agentic loop.
- Retrievers: Dense, sparse, and hybrid retrieval systems.

Usage:
    python main.py "Your query here"

Example:
    python main.py "Compare the pros and cons of Retrieval-Augmented Generation and traditional fine-tuning for language models."
"""
import yaml
import argparse
from rich.console import Console
from rich.panel import Panel
from openai import OpenAI

# Import core components
from core.router import Router
from core.planner import Planner
from core.executor import Executor
from core.verifier import Verifier
from core.aggregator import Aggregator

# Import retriever components
from retriever.dense import DenseRetriever
from retriever.sparse import BM25Retriever
from retriever.hybrid import HybridRetriever

# Import tool components
from tools.calculator import Calculator
from tools.web_search import WebSearch
from tools.database import DatabaseConnector

# Import utility components
from utils.logging import setup_logging
from utils.schema import FinalAnswer

# Initialize console for rich output
console = Console()

def load_config(config_path: str = "config.yaml") -> dict:
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def initialize_components(config: dict) -> dict:
    """Initializes all necessary components based on the configuration."""
    logger = setup_logging(log_level=config['logging']['level'], log_file=config['logging']['file'])
    logger.info("Initializing components...")

    # LLM Client
    llm_client = OpenAI(api_key=config['llm']['api_key'])

    # Mock corpus for BM25 - in a real app, this would be loaded from a data source
    # TODO: Replace this mock corpus with a real data loading mechanism.
    mock_corpus = [
        {"source": "doc1.txt", "content": "Quantum error correction is crucial for building fault-tolerant quantum computers. Surface codes are a leading approach."},
        {"source": "doc2.txt", "content": "Traditional fine-tuning updates all weights of a pre-trained model, which can be computationally expensive."},
        {"source": "doc3.txt", "content": "Retrieval-Augmented Generation (RAG) enhances LLMs by providing external knowledge, reducing hallucinations."},
        {"source": "doc4.txt", "content": "The capital of France is Paris. It is known for the Eiffel Tower."}
    ]

    # Retrievers
    # TODO: The DenseRetriever requires a pre-built FAISS index.
    # For this example, it will likely fail to initialize unless you create one.
    # We will handle its potential failure gracefully.
    try:
        dense_retriever = DenseRetriever(
            model_name=config['retriever']['dense']['model'],
            index_path=config['retriever']['dense']['index_path']
        )
    except Exception as e:
        logger.warning(f"Could not initialize DenseRetriever: {e}. It will be unavailable.")
        dense_retriever = None

    sparse_retriever = BM25Retriever(corpus=mock_corpus)

    if dense_retriever:
        hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            reranker_model_name=config['retriever']['hybrid']['reranker_model']
        )
    else:
        logger.warning("Dense retriever not available, Hybrid retriever will not be used.")
        hybrid_retriever = None


    # Tools
    tools = {}
    if config['tools']['calculator']['enabled']:
        tools['calculator'] = Calculator()
    if config['tools']['web_search']['enabled']:
        tools['web_search'] = WebSearch(api_key=config['tools']['web_search']['api_key'])
    if config['tools']['database']['enabled']:
        tools['database'] = DatabaseConnector(connection_string=config['tools']['database']['connection_string'])
    
    # Add retrievers to the list of available tools for the planner
    tools['sparse_retriever'] = sparse_retriever
    if hybrid_retriever:
        tools['hybrid_retriever'] = hybrid_retriever


    # Core agent components
    router = Router()
    planner = Planner(llm_client=llm_client, available_tools=list(tools.keys()))
    executor = Executor(tools=tools)
    verifier = Verifier(llm_client=llm_client)
    aggregator = Aggregator(llm_client=llm_client)

    logger.info("All components initialized.")
    return {
        "router": router,
        "planner": planner,
        "executor": executor,
        "verifier": verifier,
        "aggregator": aggregator,
        "sparse_retriever": sparse_retriever,
        "llm_client": llm_client,
    }

def run_simple_rag(query: str, components: dict) -> FinalAnswer:
    """
    Executes a simplified RAG pipeline for straightforward queries.
    """
    console.print("[bold cyan]Running Simple RAG Pipeline...[/bold cyan]")
    retriever = components['sparse_retriever'] # Default to sparse for simplicity
    llm_client = components['llm_client']

    # 1. Retrieve
    retrieved_docs = retriever.retrieve(query, top_k=3)
    context = "\n".join([doc.content for doc in retrieved_docs])

    # 2. Generate
    prompt = f"Based on the following context, answer the user's query.\n\nContext:\n{context}\n\nQuery: {query}"
    response = llm_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content

    return FinalAnswer(
        answer=answer,
        sources=retrieved_docs,
        confidence_score=0.6, # Default confidence for simple RAG
        unverified_claims=["Answer was generated without a verification step."]
    )

def run_agentic_pipeline(query: str, components: dict) -> FinalAnswer:
    """
    Executes the full, multi-step agentic RAG pipeline.
    """
    console.print("[bold magenta]Running Agentic RAG Pipeline...[/bold magenta]")

    # 1. Plan
    console.print("[bold]Step 1: Generating Action Plan...[/bold]")
    plan = components['planner'].generate_plan(query)
    console.print(Panel(plan.model_dump_json(indent=2), title="Action Plan", border_style="green"))

    # 2. Execute
    console.print("[bold]Step 2: Executing Plan...[/bold]")
    step_results = components['executor'].execute_plan(plan)
    # TODO: Display execution results in a structured way.

    # 3. Verify
    console.print("[bold]Step 3: Verifying Claims...[/bold]")
    # For this example, we'll just try to verify the first summary we find.
    # A real implementation would extract claims from all relevant step outputs.
    summaries = [res for res in step_results.values() if isinstance(res, str)]
    all_docs = [res for res in step_results.values() if isinstance(res, list)]
    evidence_docs = [doc for sublist in all_docs for doc in sublist]

    verification_results = []
    if summaries and evidence_docs:
        claims_to_verify = components['verifier'].extract_claims(summaries[0])
        if claims_to_verify:
            # Verify the first claim as a demo
            claim = claims_to_verify[0]
            console.print(f"Verifying claim: '{claim}'")
            ver_res = components['verifier'].verify(claim, evidence_docs)
            verification_results.append(ver_res)
            console.print(Panel(ver_res.model_dump_json(indent=2), title="Verification Result", border_style="yellow"))

    # 4. Aggregate
    console.print("[bold]Step 4: Aggregating Final Answer...[/bold]")
    final_answer = components['aggregator'].synthesize_answer(
        query=query,
        step_results=step_results,
        verification_results=verification_results
    )
    return final_answer

def main():
    """Main function to run the CLI."""
    parser = argparse.ArgumentParser(description="Agentic RAG Pipeline CLI")
    parser.add_argument("query", type=str, help="The query to process.")
    args = parser.parse_args()

    console.print(Panel(f"[bold]Query:[/bold] {args.query}", title="[bold blue]Agentic RAG System[/bold blue]", border_style="blue"))

    try:
        config = load_config()
        components = initialize_components(config)

        # Use the router to decide the path
        path = components['router'].decide_path(args.query)

        if path == "simple_rag":
            final_answer = run_simple_rag(args.query, components)
        else:
            final_answer = run_agentic_pipeline(args.query, components)

        # Print the final answer
        console.print(Panel(final_answer.answer, title="[bold green]Final Answer[/bold green]", border_style="green"))
        console.print(f"[bold]Confidence:[/bold] {final_answer.confidence_score:.2f}")

        if final_answer.sources:
            console.print("[bold]Sources:[/bold]")
            for i, source in enumerate(final_answer.sources):
                console.print(f"  {i+1}. {source.source} (Score: {source.score:.2f})")
        
        if final_answer.unverified_claims:
            console.print("[bold yellow]Unverified Claims:[/bold yellow]")
            for claim in final_answer.unverified_claims:
                console.print(f"  - {claim}")

    except FileNotFoundError:
        console.print("[bold red]Error: config.yaml not found. Please ensure it exists in the root directory.[/bold red]")
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        # Also log the full traceback for debugging
        logger = setup_logging()
        logger.exception("An unhandled exception occurred in main.")

if __name__ == "__main__":
    main()
