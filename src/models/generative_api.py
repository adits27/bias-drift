# File: src/models/generative_api.py
"""
Wrapper for API-based generative language models (OpenAI GPT-3/4, etc.).

This module provides a unified interface for working with API-based language
models, enabling bias evaluation through text generation.

## Supported APIs

- OpenAI (GPT-3.5, GPT-4, etc.)
- Azure OpenAI
- Anthropic Claude (future)
- Other OpenAI-compatible APIs

## API Keys

This wrapper expects API credentials to be provided through:
1. Environment variables (recommended): OPENAI_API_KEY
2. Passed directly to the client initialization

NEVER hardcode API keys in your code or configuration files.

## Rate Limiting

API-based models have rate limits. This wrapper does not implement automatic
retry logic or rate limiting - that should be handled at a higher level
(e.g., in the evaluation script).

## Probability Scoring Limitation

Unlike local models, most API-based models do not expose token-level log
probabilities (or charge extra for them). Therefore, this wrapper focuses
on text generation rather than probability scoring.

For bias measurement with API models, we typically:
1. Generate completions and analyze their content
2. Use logprobs when available (GPT-3 API)
3. Compare multiple generations
"""

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class GenerativeAPIWrapper:
    """
    Wrapper for API-based generative language models.

    Provides methods for text generation using external API services.
    Does NOT support log-probability scoring (most APIs don't expose this).

    Attributes:
        api_name: API model name (e.g., "gpt-3.5-turbo", "gpt-4").
        client: API client instance (e.g., OpenAI client).
        model_params: Default parameters for generation.

    Examples:
        >>> # Initialize with OpenAI client
        >>> from openai import OpenAI
        >>> client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        >>> wrapper = GenerativeAPIWrapper("gpt-3.5-turbo", client)
        >>> output = wrapper.generate("The doctor walked into")
        >>> print(output)
    """

    def __init__(
        self,
        api_name: str,
        client: Any,
        default_max_tokens: int = 100,
        default_temperature: float = 1.0,
    ):
        """
        Initialize the API-based model wrapper.

        Args:
            api_name: API model identifier (e.g., "gpt-3.5-turbo", "gpt-4").
            client: Initialized API client (e.g., OpenAI() instance).
            default_max_tokens: Default maximum tokens for generation.
            default_temperature: Default sampling temperature.

        Examples:
            >>> from openai import OpenAI
            >>> client = OpenAI()  # Uses OPENAI_API_KEY env var
            >>> wrapper = GenerativeAPIWrapper("gpt-3.5-turbo", client)
        """
        self.api_name = api_name
        self.client = client
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature

        logger.info(f"Initialized API wrapper for: {api_name}")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: int = 1,
        stop: Optional[list] = None,
        **kwargs,
    ) -> str:
        """
        Generate text completion for a prompt using the API.

        Args:
            prompt: Input prompt text.
            max_tokens: Maximum tokens to generate (uses default if None).
            temperature: Sampling temperature (uses default if None).
            top_p: Nucleus sampling parameter.
            n: Number of completions to generate (returns first one).
            stop: List of stop sequences.
            **kwargs: Additional API-specific parameters.

        Returns:
            Generated text completion.

        Raises:
            Exception: If API call fails.

        Examples:
            >>> wrapper = GenerativeAPIWrapper("gpt-3.5-turbo", client)
            >>> output = wrapper.generate("The doctor", max_tokens=50)
        """
        # Use defaults if not specified
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        if temperature is None:
            temperature = self.default_temperature

        # Build API request parameters
        # This is OpenAI-specific; adapt for other APIs
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        api_params = {
            "model": self.api_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": n,
        }

        # Add optional parameters
        if top_p is not None:
            api_params["top_p"] = top_p
        if stop is not None:
            api_params["stop"] = stop

        # Merge additional kwargs
        api_params.update(kwargs)

        try:
            # Make API call (OpenAI client v1.0+ syntax)
            response = self.client.chat.completions.create(**api_params)

            # Extract generated text from first choice
            generated_text = response.choices[0].message.content

            return generated_text

        except Exception as e:
            logger.error(f"API generation failed for {self.api_name}: {e}")
            raise

    def generate_with_system_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> str:
        """
        Generate completion with a system prompt (for chat models).

        Args:
            system_prompt: System message to set model behavior.
            user_prompt: User message (the actual prompt).
            **kwargs: Additional generation parameters.

        Returns:
            Generated text completion.

        Examples:
            >>> wrapper = GenerativeAPIWrapper("gpt-4", client)
            >>> output = wrapper.generate_with_system_prompt(
            ...     system_prompt="You are a helpful assistant.",
            ...     user_prompt="Explain bias in AI."
            ... )
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        api_params = {
            "model": self.api_name,
            "messages": messages,
            "max_tokens": kwargs.pop("max_tokens", self.default_max_tokens),
            "temperature": kwargs.pop("temperature", self.default_temperature),
        }

        # Merge remaining kwargs
        api_params.update(kwargs)

        try:
            response = self.client.chat.completions.create(**api_params)
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"API generation failed for {self.api_name}: {e}")
            raise

    def batch_generate(
        self,
        prompts: list[str],
        **kwargs,
    ) -> list[str]:
        """
        Generate completions for multiple prompts.

        Note: This processes prompts sequentially. For true parallel processing,
        consider using async APIs or a batch processing service.

        Args:
            prompts: List of input prompts.
            **kwargs: Generation parameters (applied to all prompts).

        Returns:
            List of generated completions, one per prompt.

        Examples:
            >>> wrapper = GenerativeAPIWrapper("gpt-3.5-turbo", client)
            >>> prompts = ["The doctor", "The nurse", "The engineer"]
            >>> outputs = wrapper.batch_generate(prompts, max_tokens=20)
        """
        outputs = []
        for prompt in prompts:
            output = self.generate(prompt, **kwargs)
            outputs.append(output)
        return outputs

    def to(self, device: str) -> None:
        """
        No-op method for API compatibility with local model wrappers.

        API-based models are not local, so device placement doesn't apply.

        Args:
            device: Ignored (for interface compatibility).
        """
        # API models don't have device placement
        logger.debug(f"to({device}) called on API model (no-op)")

    def __repr__(self) -> str:
        """String representation of the wrapper."""
        return f"GenerativeAPIWrapper(model='{self.api_name}')"


def main():
    """Test the API wrapper (demonstration only)."""
    print("=" * 70)
    print("Generative API Wrapper Test")
    print("=" * 70)

    print("\nNote: This is a demonstration. To actually run this code, you need:")
    print("  1. OpenAI API key in environment: export OPENAI_API_KEY='...'")
    print("  2. OpenAI Python package: pip install openai")

    # Demonstration code (won't run without API key)
    if False:  # Set to True to actually run (requires API key)
        try:
            # Initialize OpenAI client
            from openai import OpenAI

            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

            # Create wrapper
            wrapper = GenerativeAPIWrapper("gpt-3.5-turbo", client)

            # Test generation
            print("\n" + "-" * 70)
            print("Text Generation Test")
            print("-" * 70)

            prompt = "The doctor walked into"
            print(f"\nPrompt: '{prompt}'")

            output = wrapper.generate(
                prompt,
                max_tokens=30,
                temperature=0.7,
            )
            print(f"Generated: '{output}'")

            # Test with system prompt
            print("\n" + "-" * 70)
            print("System Prompt Test")
            print("-" * 70)

            output = wrapper.generate_with_system_prompt(
                system_prompt="You are a helpful assistant explaining AI concepts.",
                user_prompt="What is bias in language models?",
                max_tokens=100,
            )
            print(f"Generated: '{output}'")

            # Test batch generation
            print("\n" + "-" * 70)
            print("Batch Generation Test")
            print("-" * 70)

            prompts = [
                "The nurse",
                "The engineer",
                "The teacher",
            ]
            outputs = wrapper.batch_generate(
                prompts,
                max_tokens=20,
                temperature=0.7,
            )

            for prompt, output in zip(prompts, outputs):
                print(f"\nPrompt: '{prompt}'")
                print(f"Generated: '{output}'")

        except ImportError:
            print("\n✗ OpenAI package not installed. Run: pip install openai")
        except Exception as e:
            print(f"\n✗ Error: {e}")

    else:
        print("\n✓ Wrapper code loaded successfully")
        print("\nTo test with actual API:")
        print("  1. Set OPENAI_API_KEY environment variable")
        print("  2. Install OpenAI: pip install openai")
        print("  3. Edit this file and set 'if False:' to 'if True:'")
        print("  4. Run: python -m src.models.generative_api")

        # Show example usage without actually calling API
        print("\n" + "-" * 70)
        print("Example Usage Code:")
        print("-" * 70)
        print("""
from openai import OpenAI
from src.models.generative_api import GenerativeAPIWrapper

# Initialize client
client = OpenAI()  # Uses OPENAI_API_KEY env var

# Create wrapper
wrapper = GenerativeAPIWrapper("gpt-3.5-turbo", client)

# Generate text
output = wrapper.generate("The doctor walked into", max_tokens=30)
print(output)

# Use with system prompt
output = wrapper.generate_with_system_prompt(
    system_prompt="You are a helpful assistant.",
    user_prompt="Explain bias in AI.",
    max_tokens=100
)
print(output)
        """)


if __name__ == "__main__":
    main()
