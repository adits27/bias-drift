# File: src/models/masked_lm.py
"""
Wrapper for masked language models (BERT, RoBERTa, etc.).

This module provides a unified interface for working with masked language models
from HuggingFace, enabling bias evaluation through pseudo-log-likelihood scoring.

## Pseudo-Log-Likelihood Estimation

Masked LMs don't directly provide sentence probabilities. We use the pseudo-log-
likelihood (PLL) method from Salazar et al. (2020):

1. For each token position i in the sentence:
   - Mask token i
   - Get model's predicted probability for the true token at position i
   - Take log probability
2. Sum log probabilities across all positions

This approximates the likelihood of the sentence under the model.

Reference:
Salazar et al. (2020). Masked Language Model Scoring.
ACL 2020. https://aclanthology.org/2020.acl-main.240/
"""

import logging
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

logger = logging.getLogger(__name__)


class MaskedLMWrapper:
    """
    Wrapper for masked language models (BERT, RoBERTa, etc.).

    Provides methods for computing pseudo-log-likelihood scores for sentences,
    which can be used to measure stereotypical associations in bias benchmarks.

    Attributes:
        model_name: HuggingFace model identifier.
        tokenizer: HuggingFace tokenizer instance.
        model: HuggingFace masked LM instance.
        device: Device for model inference (cpu, cuda, mps).
        mask_token_id: Token ID for the [MASK] token.

    Examples:
        >>> wrapper = MaskedLMWrapper("bert-base-uncased", device="cuda")
        >>> score = wrapper.sentence_logprob("The cat sat on the mat.")
        >>> print(f"Log-probability: {score:.4f}")
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        use_fp16: bool = False,
    ):
        """
        Initialize the masked language model wrapper.

        Args:
            model_name: HuggingFace model identifier (e.g., "bert-base-uncased").
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

        logger.info(f"Loading masked LM: {model_name} on {device}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Load model
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)

            # Move to device
            self.model.to(self.device)

            # Set to evaluation mode
            self.model.eval()

            # Convert to FP16 if requested and supported
            if use_fp16 and device == "cuda":
                self.model.half()

            # Get mask token ID
            self.mask_token_id = self.tokenizer.mask_token_id

            if self.mask_token_id is None:
                raise ValueError(
                    f"Model {model_name} does not have a mask token. "
                    f"This wrapper only supports masked language models."
                )

            logger.info(f"Successfully loaded {model_name}")

        except Exception as e:
            raise ValueError(f"Error loading model {model_name}: {e}")

    def sentence_logprob(self, text: str) -> float:
        """
        Compute the pseudo-log-likelihood of a sentence.

        Uses the masked language model scoring method: for each token position,
        mask that token and compute the model's log-probability of the true token.
        Sum across all positions.

        Args:
            text: Input sentence.

        Returns:
            Total pseudo-log-likelihood (sum of per-token log probabilities).

        Examples:
            >>> wrapper = MaskedLMWrapper("bert-base-uncased")
            >>> score1 = wrapper.sentence_logprob("The doctor helped the nurse.")
            >>> score2 = wrapper.sentence_logprob("The nurse helped the doctor.")
            >>> # Compare scores to measure bias
        """
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        # Get valid token positions (exclude special tokens like [CLS], [SEP])
        # We'll score all tokens except special tokens
        special_token_ids = set(self.tokenizer.all_special_ids)

        total_logprob = 0.0
        num_scored_tokens = 0

        with torch.no_grad():
            # Iterate over each token position
            for position in range(input_ids.size(1)):
                token_id = input_ids[0, position].item()

                # Skip special tokens
                if token_id in special_token_ids:
                    continue

                # Create masked input
                masked_input_ids = input_ids.clone()
                masked_input_ids[0, position] = self.mask_token_id

                # Forward pass
                outputs = self.model(masked_input_ids)
                logits = outputs.logits

                # Get logits for the masked position
                position_logits = logits[0, position]  # Shape: [vocab_size]

                # Compute log probabilities
                log_probs = F.log_softmax(position_logits, dim=-1)

                # Get log probability of the true token
                token_logprob = log_probs[token_id].item()

                total_logprob += token_logprob
                num_scored_tokens += 1

        # Return total log probability
        # Note: This is NOT normalized by sentence length
        # For bias measurement, we typically compare sentences of similar length
        return total_logprob

    def compare_sentences(
        self,
        sentence1: str,
        sentence2: str,
    ) -> dict:
        """
        Compare two sentences by their pseudo-log-likelihoods.

        This is useful for bias measurement tasks like CrowS-Pairs, where
        we compare stereotypical vs. anti-stereotypical sentence pairs.

        Args:
            sentence1: First sentence.
            sentence2: Second sentence.

        Returns:
            Dictionary containing:
            - logprob_1: Log-probability of sentence 1
            - logprob_2: Log-probability of sentence 2
            - preferred: Which sentence has higher probability (1 or 2)
            - diff: Difference in log-probabilities (logprob_1 - logprob_2)

        Examples:
            >>> wrapper = MaskedLMWrapper("bert-base-uncased")
            >>> result = wrapper.compare_sentences(
            ...     "The doctor helped the nurse.",
            ...     "The nurse helped the doctor."
            ... )
            >>> print(f"Preferred: sentence {result['preferred']}")
        """
        logprob_1 = self.sentence_logprob(sentence1)
        logprob_2 = self.sentence_logprob(sentence2)

        diff = logprob_1 - logprob_2
        preferred = 1 if logprob_1 > logprob_2 else 2

        return {
            "logprob_1": logprob_1,
            "logprob_2": logprob_2,
            "preferred": preferred,
            "diff": diff,
        }

    def batch_sentence_logprobs(
        self,
        texts: List[str],
        batch_size: int = 8,
    ) -> List[float]:
        """
        Compute pseudo-log-likelihoods for a batch of sentences.

        Note: Current implementation processes sentences individually.
        True batching would require padding to equal length.

        Args:
            texts: List of input sentences.
            batch_size: Batch size (currently unused, for future optimization).

        Returns:
            List of log-probabilities, one per sentence.

        Examples:
            >>> wrapper = MaskedLMWrapper("bert-base-uncased")
            >>> sentences = ["The cat sat.", "The dog ran.", "The bird flew."]
            >>> scores = wrapper.batch_sentence_logprobs(sentences)
        """
        # TODO: Implement true batching with padding for efficiency
        scores = []
        for text in texts:
            score = self.sentence_logprob(text)
            scores.append(score)
        return scores

    def to(self, device: str) -> None:
        """
        Move model to a different device.

        Args:
            device: Target device ("cpu", "cuda", "mps").

        Examples:
            >>> wrapper = MaskedLMWrapper("bert-base-uncased", device="cpu")
            >>> wrapper.to("cuda")
        """
        self.device = device
        self.model.to(device)
        logger.info(f"Moved model to {device}")

    def __repr__(self) -> str:
        """String representation of the wrapper."""
        return (
            f"MaskedLMWrapper(model='{self.model_name}', "
            f"device='{self.device}', fp16={self.use_fp16})"
        )


def main():
    """Test the masked LM wrapper."""
    print("=" * 70)
    print("Masked LM Wrapper Test")
    print("=" * 70)

    # Load BERT
    print("\nLoading bert-base-uncased...")
    wrapper = MaskedLMWrapper("bert-base-uncased", device="cpu")

    # Test sentence scoring
    test_sentence = "The cat sat on the mat."
    print(f"\nTest sentence: '{test_sentence}'")

    score = wrapper.sentence_logprob(test_sentence)
    print(f"Pseudo-log-likelihood: {score:.4f}")

    # Test sentence comparison
    print("\n" + "-" * 70)
    print("Sentence Comparison Test (Gender Bias)")
    print("-" * 70)

    sent1 = "The doctor asked the nurse to help him."
    sent2 = "The doctor asked the nurse to help her."

    print(f"\nSentence 1 (stereotypical): '{sent1}'")
    print(f"Sentence 2 (anti-stereotypical): '{sent2}'")

    result = wrapper.compare_sentences(sent1, sent2)
    print(f"\nResults:")
    print(f"  Sentence 1 log-prob: {result['logprob_1']:.4f}")
    print(f"  Sentence 2 log-prob: {result['logprob_2']:.4f}")
    print(f"  Difference: {result['diff']:.4f}")
    print(f"  Preferred: Sentence {result['preferred']}")

    if result['preferred'] == 1:
        print("\n⚠ Model assigns higher probability to stereotypical sentence")
    else:
        print("\n✓ Model assigns higher probability to anti-stereotypical sentence")


if __name__ == "__main__":
    main()
