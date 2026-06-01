"""
LangChain + Mellea Integration Demo
Generates product descriptions with automatic validation and retries.
"""

from mellea import start_session
from mellea_langchain import MelleaChatModel
from mellea.stdlib.requirements import req, simple_validate, Requirement
from mellea.stdlib.sampling import RejectionSamplingStrategy
from langchain_core.prompts import ChatPromptTemplate
import time
import sys

def print_slow(text, delay=0.3):
    """Print text line-by-line with delay for video visibility."""
    for line in text.split('\n'):
        print(line)
        sys.stdout.flush()
        time.sleep(delay)

word_count_req = Requirement(
    description="Answer must be between 50-100 words",
    validation_fn=simple_validate(
        lambda x: (50 <= len(x.split()) <= 100,
                   f"Output is {len(x.split())} words; must be 50-100.")
    )
)

def demo_product_description():
    """Demo: Generate product descriptions with Mellea validation."""

    # Initialize Mellea session
    m = start_session(backend_name="ollama", model_id="llama3.2:3b")
    chat_model = MelleaChatModel(
        mellea_session=m,
        model="mellea-ollama",
        model_id="llama3.2:3b"
    )

    # Define the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that writes compelling product descriptions."),
        ("human", "Write a product description for: {product}")
    ])

    # Validated chain with quality guarantees
    chain = prompt | chat_model.bind(
        model_options={
            "requirements": [
                word_count_req,
                req("Must mention at least two product features"),
                req("Must include a clear benefit statement"),
            ],
            "strategy": RejectionSamplingStrategy(loop_budget=3),
        }
    )

    # Run the chain
    products = [
        "Wireless Bluetooth Headphones"
    ]

    for product in products:
        print(f"\n{'='*60}")
        print(f"Generating: {product}")
        print(f"{'='*60}")

        start_time = time.time()

        try:
            result = chain.invoke({"product": product})
            elapsed = time.time() - start_time

            answer = result.content if hasattr(result, 'content') else result
            word_count = len(answer.split())

            print(f"\nResult:")
            print_slow(answer, delay=0.2)
            print(f"\nWord count: {word_count}")
            print(f"✓ Passed validation in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n✗ Failed: {e}")
            print(f"Time spent: {elapsed:.1f}s")

        print()


if __name__ == "__main__":
    print("LangChain + Mellea Integration Demo")
    print("=" * 60)

    # Run product description demo
    demo_product_description()
