# File: src/models/generative_hf.py
"""
Wrapper for HuggingFace generative language models (GPT-2, GPT-Neo, etc.).

This module provides a unified interface for working with autoregressive/causal
language models from HuggingFace, enabling both text generation and probability
scoring for bias evaluation.

## Use Cases

1. **Text Generation**: Generate continuations for prompts to measure biased outputs
2. **Probability Scoring**: Compute log-probabilities of continuations to compare
   stereotypical vs. anti-stereotypical completions

## Models Supported

- GPT-2 (all sizes)
- GPT-Neo
- GPT-J
- OPT
- LLaMA (if accessible)
- Any HuggingFace causal LM
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class GenerativeHFWrapper:
    """
    Wrapper for HuggingFace generative language models.

    Provides methods for text generation and log-probability computation,
    enabling bias measurement in autoregressive language models.

    Attributes:
        model_name: HuggingFace model identifier.
        tokenizer: HuggingFace tokenizer instance.
        model: HuggingFace causal LM instance.
        device: Device for model inference.

    Examples:
        >>> wrapper = GenerativeHFWrapper("gpt2", device="cuda")
        >>> # Generate text
        >>> output = wrapper.generate("The doctor walked into", max_new_tokens=10)
        >>> print(output)
        >>> # Score a continuation
        >>> score = wrapper.logprob_of_continuation("The doctor", "walked into the room")
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        use_fp16: bool = False,
    ):
        """
        Initialize the generative language model wrapper.

        Args:
            model_name: HuggingFace model identifier (e.g., "gpt2").
            device: Device for inference ("cpu", "cuda", "mps", or "auto").
            use_fp16: Whether to use FP16 precision for faster inference.

        Raises:
            ValueError: If the model cannot be loaded.
        """
        self.model_name = model_name
        self.use_fp16 = use_fp16

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device

        logger.info(f"Loading generative HF model: {model_name} on {device}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Set pad token if not already set (GPT-2 doesn't have one by default)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

            # Move to device
            self.model.to(self.device)

            # Set to evaluation mode
            self.model.eval()

            # Convert to FP16 if requested and supported
            if use_fp16 and device == "cuda":
                self.model.half()

            logger.info(f"Successfully loaded {model_name}")

        except Exception as e:
            raise ValueError(f"Error loading model {model_name}: {e}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> str:
        """
        Generate text continuation for a prompt.

        Args:
            prompt: Input prompt text.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_p: Nucleus sampling threshold (if None, not used).
            top_k: Top-k sampling threshold (if None, not used).
            num_return_sequences: Number of sequences to generate.
            **kwargs: Additional arguments passed to model.generate().

        Returns:
            Generated text (continuation only, without the prompt).
            If num_return_sequences > 1, returns the first sequence.

        Examples:
            >>> wrapper = GenerativeHFWrapper("gpt2")
            >>> output = wrapper.generate("The doctor", max_new_tokens=10)
            >>> print(output)
        """
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        prompt_length = input_ids.size(1)

        # Set up generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True if temperature > 0 else False,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        # Add top_p and top_k if specified
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if top_k is not None:
            gen_kwargs["top_k"] = top_k

        # Merge with additional kwargs
        gen_kwargs.update(kwargs)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(input_ids, **gen_kwargs)

        # Decode only the generated portion (not the prompt)
        # outputs shape: [num_return_sequences, total_length]
        generated_ids = outputs[0, prompt_length:]  # Take first sequence, skip prompt
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text

    def logprob_of_continuation(
        self,
        prompt: str,
        continuation: str,
    ) -> float:
        """
        Compute the log-probability of a continuation given a prompt.

        This is useful for bias measurement: compare the probability of
        stereotypical vs. anti-stereotypical continuations.

        Args:
            prompt: The context/prompt text.
            continuation: The continuation text to score.

        Returns:
            Total log-probability of the continuation tokens given the prompt.

        Examples:
            >>> wrapper = GenerativeHFWrapper("gpt2")
            >>> # Compare two continuations
            >>> prompt = "The nurse"
            >>> cont1 = " helped her patient"
            >>> cont2 = " helped his patient"
            >>> score1 = wrapper.logprob_of_continuation(prompt, cont1)
            >>> score2 = wrapper.logprob_of_continuation(prompt, cont2)
            >>> if score1 > score2:
            ...     print("Model prefers feminine pronoun for 'nurse'")
        """
        # Tokenize prompt and full text separately to identify continuation tokens
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt")
        prompt_length = prompt_tokens["input_ids"].size(1)

        # Tokenize full text (prompt + continuation)
        full_text = prompt + continuation
        full_tokens = self.tokenizer(full_text, return_tensors="pt")
        input_ids = full_tokens["input_ids"].to(self.device)

        # Get model outputs
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits  # Shape: [1, seq_len, vocab_size]

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Sum log probabilities for continuation tokens
        # Note: logits[0, i] predicts token at position i+1
        total_logprob = 0.0
        num_continuation_tokens = 0

        for i in range(prompt_length, input_ids.size(1)):
            # Get predicted token ID (ground truth)
            token_id = input_ids[0, i]

            # Get log probability from previous position's predictions
            if i > 0:
                token_logprob = log_probs[0, i - 1, token_id].item()
                total_logprob += token_logprob
                num_continuation_tokens += 1

        return total_logprob

    def compare_continuations(
        self,
        prompt: str,
        continuation1: str,
        continuation2: str,
    ) -> dict:
        """
        Compare two continuations by their log-probabilities.

        Useful for bias measurement tasks where we compare stereotypical
        vs. anti-stereotypical completions.

        Args:
            prompt: The context/prompt text.
            continuation1: First continuation.
            continuation2: Second continuation.

        Returns:
            Dictionary containing:
            - logprob_1: Log-probability of continuation 1
            - logprob_2: Log-probability of continuation 2
            - preferred: Which continuation has higher probability (1 or 2)
            - diff: Difference in log-probabilities (logprob_1 - logprob_2)

        Examples:
            >>> wrapper = GenerativeHFWrapper("gpt2")
            >>> result = wrapper.compare_continuations(
            ...     "The nurse",
            ...     " helped her patient",
            ...     " helped his patient"
            ... )
            >>> print(f"Preferred: continuation {result['preferred']}")
        """
        logprob_1 = self.logprob_of_continuation(prompt, continuation1)
        logprob_2 = self.logprob_of_continuation(prompt, continuation2)

        diff = logprob_1 - logprob_2
        preferred = 1 if logprob_1 > logprob_2 else 2

        return {
            "logprob_1": logprob_1,
            "logprob_2": logprob_2,
            "preferred": preferred,
            "diff": diff,
        }

    def to(self, device: str) -> None:
        """
        Move model to a different device.

        Args:
            device: Target device ("cpu", "cuda", "mps").

        Examples:
            >>> wrapper = GenerativeHFWrapper("gpt2", device="cpu")
            >>> wrapper.to("cuda")
        """
        self.device = device
        self.model.to(device)
        logger.info(f"Moved model to {device}")

    def __repr__(self) -> str:
        """String representation of the wrapper."""
        return (
            f"GenerativeHFWrapper(model='{self.model_name}', "
            f"device='{self.device}', fp16={self.use_fp16})"
        )


def main():
    """Test the generative HF wrapper."""
    print("=" * 70)
    print("Generative HF Wrapper Test")
    print("=" * 70)

    # Load GPT-2
    print("\nLoading gpt2...")
    wrapper = GenerativeHFWrapper("gpt2", device="cpu")

    # Test text generation
    print("\n" + "-" * 70)
    print("Text Generation Test")
    print("-" * 70)

    prompt = "The doctor walked into"
    print(f"\nPrompt: '{prompt}'")

    generated = wrapper.generate(
        prompt,
        max_new_tokens=20,
        temperature=0.7,
        top_p=0.9
    )
    print(f"Generated: '{generated}'")

    # Test continuation scoring
    print("\n" + "-" * 70)
    print("Continuation Scoring Test (Gender Bias)")
    print("-" * 70)

    prompt = "The nurse told the patient that"
    cont1 = " she would be back soon"
    cont2 = " he would be back soon"

    print(f"\nPrompt: '{prompt}'")
    print(f"Continuation 1 (feminine): '{cont1}'")
    print(f"Continuation 2 (masculine): '{cont2}'")

    result = wrapper.compare_continuations(prompt, cont1, cont2)
    print(f"\nResults:")
    print(f"  Continuation 1 log-prob: {result['logprob_1']:.4f}")
    print(f"  Continuation 2 log-prob: {result['logprob_2']:.4f}")
    print(f"  Difference: {result['diff']:.4f}")
    print(f"  Preferred: Continuation {result['preferred']}")

    if result['preferred'] == 1:
        print("\n⚠ Model assigns higher probability to feminine pronoun for 'nurse'")
    else:
        print("\n✓ Model assigns higher probability to masculine pronoun for 'nurse'")

    # Test direct logprob computation
    print("\n" + "-" * 70)
    print("Direct Log-Probability Test")
    print("-" * 70)

    test_prompt = "The sky is"
    test_continuation = " blue"
    score = wrapper.logprob_of_continuation(test_prompt, test_continuation)
    print(f"\nPrompt: '{test_prompt}'")
    print(f"Continuation: '{test_continuation}'")
    print(f"Log-probability: {score:.4f}")


if __name__ == "__main__":
    main()
