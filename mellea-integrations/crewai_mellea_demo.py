"""
CrewAI + Mellea Integration Demo
Multi-agent system with independent validation per agent.
"""

from mellea import start_session
from mellea_crewai import MelleaLLM
from mellea.stdlib.requirements import req, simple_validate, Requirement
from mellea.stdlib.sampling import RejectionSamplingStrategy
from crewai import Agent, Task, Crew
import time
import sys

def print_slow(text, delay=0.3):
    """Print text line-by-line with delay for video visibility."""
    for line in text.split('\n'):
        print(line)
        sys.stdout.flush()
        time.sleep(delay)

word_count_req = Requirement(
    description="Answer must be between 50-200 words",
    validation_fn=simple_validate(
        lambda x: (50 <= len(x.split()) <= 200,
                   f"Output is {len(x.split())} words; must be 50-200.")
    )
)


def demo_multi_agent_validation():
    """Demo: Multi-agent system with independent validation per agent."""

    print("="*60)
    print("CrewAI + Mellea Multi-Agent Demo")
    print("="*60)

    m = start_session(backend_name="ollama", model_id="llama3.2:3b")

    # Agent A: Summary Writer
    # Requirements: under 200 words, explain the topic (easier to pass on first try)
    summary_agent = Agent(
        role="Summary Writer",
        goal="Write concise and clear summaries of technical topics",
        backstory="You are an expert at distilling complex information into clear, brief summaries.",
        llm=MelleaLLM(
            mellea_session=m,
            model="mellea-ollama",
            model_id="llama3.2:3b",
            requirements=[
                word_count_req,
                req("Must explain the topic clearly"),
            ],
            strategy=RejectionSamplingStrategy(loop_budget=3),
        )
    )

    # Agent B: JSON Formatter
    # Requirements: valid JSON, specific fields, arrays with minimum items, short point descriptions
    formatter_agent = Agent(
        role="JSON Formatter",
        goal="Format information as structured JSON",
        backstory="You are an expert at organizing data into clean, valid JSON structures.",
        llm=MelleaLLM(
            mellea_session=m,
            model="mellea-ollama",
            model_id="llama3.2:3b",
            requirements=[
                req("Output must be valid JSON"),
                req("Must include 'topic' field with value 'Machine Learning'"),
                req("Must include 'keyPoints' as an array with exactly 3 items"),
                req("Each keyPoint description must be under 3 words"),
            ],
            strategy=RejectionSamplingStrategy(loop_budget=3),
        )
    )

    # Task 1: Summary generation
    summary_task = Task(
        description="Summarize the following topic: Machine Learning in Production. "
                    "Explain what it is, why it matters, and mention a key challenge.",
        expected_output="A brief summary (under 200 words) of machine learning in production",
        agent=summary_agent
    )

    # Task 2: JSON formatting
    json_task = Task(
        description="Format the following as valid JSON: "
                    "Topic: Machine Learning, Key Points: Model training, deployment, monitoring, performance tracking.",
        expected_output="Valid JSON with topic and keyPoints array",
        agent=formatter_agent
    )

    # Create crew
    crew = Crew(
        agents=[summary_agent, formatter_agent],
        tasks=[summary_task, json_task],
        verbose=False,
    )

    print("\nStarting multi-agent crew with independent validation...\n")
    start_time = time.time()

    try:
        result = crew.kickoff()
        elapsed = time.time() - start_time

        print(f"\n{'='*60}")
        print("Multi-Agent Validation Complete")
        print(f"{'='*60}")
        print(f"\nResults:")
        print_slow(str(result), delay=0.2)
        print(f"\n✓ All agents passed validation in {elapsed:.1f}s")

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Validation failed: {e}")
        print(f"Time spent: {elapsed:.1f}s")
if __name__ == "__main__":
    # Run multi-agent demo with different validation per agent
    demo_multi_agent_validation()

