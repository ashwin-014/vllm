"""Sampling parameters for text generation."""
from typing import List, Optional, Union, Dict, Any
from abc import ABC
import torch

_SAMPLING_EPS = 1e-5


class SamplingParams:
    """Sampling parameters for text generation.

    Overall, we follow the sampling parameters from the OpenAI text completion
    API (https://platform.openai.com/docs/api-reference/completions/create).
    In addition, we support beam search, which is not supported by OpenAI.

    Args:
        n: Number of output sequences to return for the given prompt.
        best_of: Number of output sequences that are generated from the prompt.
            From these `best_of` sequences, the top `n` sequences are returned.
            `best_of` must be greater than or equal to `n`. This is treated as
            the beam width when `use_beam_search` is True. By default, `best_of`
            is set to `n`.
        presence_penalty: Float that penalizes new tokens based on whether they
            appear in the generated text so far. Values > 0 encourage the model
            to use new tokens, while values < 0 encourage the model to repeat
            tokens.
        frequency_penalty: Float that penalizes new tokens based on their
            frequency in the generated text so far. Values > 0 encourage the
            model to use new tokens, while values < 0 encourage the model to
            repeat tokens.
        temperature: Float that controls the randomness of the sampling. Lower
            values make the model more deterministic, while higher values make
            the model more random. Zero means greedy sampling.
        top_p: Float that controls the cumulative probability of the top tokens
            to consider. Must be in (0, 1]. Set to 1 to consider all tokens.
        top_k: Integer that controls the number of top tokens to consider. Set
            to -1 to consider all tokens.
        use_beam_search: Whether to use beam search instead of sampling.
        stop: List of strings that stop the generation when they are generated.
            The returned output will not contain the stop strings.
        ignore_eos: Whether to ignore the EOS token and continue generating
            tokens after the EOS token is generated.
        max_tokens: Maximum number of tokens to generate per output sequence.
        logprobs: Number of log probabilities to return per output token.
    """

    def __init__(
        self,
        n: int = 1,
        best_of: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        use_beam_search: bool = False,
        stop: Union[None, str, List[str]] = None,
        ignore_eos: bool = False,
        max_tokens: int = 16,
        logprobs: Optional[int] = None,
        output_guidance_config: Optional[Dict[str, Any]] = {},
    ) -> None:
        self.n = n
        self.best_of = best_of if best_of is not None else n
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.use_beam_search = use_beam_search
        if stop is None:
            self.stop = []
        elif isinstance(stop, str):
            self.stop = [stop]
        else:
            self.stop = list(stop)
        self.ignore_eos = ignore_eos
        self.max_tokens = max_tokens
        self.logprobs = logprobs

        self.logits_warper = None
        warper_name = output_guidance_config.get("logits_warper", None)
        if warper_name == "selection":
            self.logits_warper = Selection(**output_guidance_config)

        self._verify_args()
        if self.use_beam_search:
            self._verity_beam_search()
        elif self.temperature < _SAMPLING_EPS:
            # Zero temperature means greedy sampling.
            self._verify_greedy_sampling()

    def maintain_seq_state(self, seq_id: int, seq_state: List[Any]) -> None:
        self.sequence_state[seq_id] = seq_state

    def _verify_args(self) -> None:
        if self.n < 1:
            raise ValueError(f"n must be at least 1, got {self.n}.")
        if self.best_of < self.n:
            raise ValueError(f"best_of must be greater than or equal to n, "
                             f"got n={self.n} and best_of={self.best_of}.")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError("presence_penalty must be in [-2, 2], got "
                             f"{self.presence_penalty}.")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError("frequency_penalty must be in [-2, 2], got "
                             f"{self.frequency_penalty}.")
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}.")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(f"top_k must be -1 (disable), or at least 1, "
                             f"got {self.top_k}.")
        if self.max_tokens < 1:
            raise ValueError(
                f"max_tokens must be at least 1, got {self.max_tokens}.")
        if self.logprobs is not None and self.logprobs < 0:
            raise ValueError(
                f"logprobs must be non-negative, got {self.logprobs}.")

    def _verity_beam_search(self) -> None:
        if self.best_of == 1:
            raise ValueError("best_of must be greater than 1 when using beam "
                             f"search. Got {self.best_of}.")
        if self.temperature > _SAMPLING_EPS:
            raise ValueError("temperature must be 0 when using beam search.")
        if self.top_p < 1.0 - _SAMPLING_EPS:
            raise ValueError("top_p must be 1 when using beam search.")
        if self.top_k != -1:
            raise ValueError("top_k must be -1 when using beam search.")

    def _verify_greedy_sampling(self) -> None:
        if self.best_of > 1:
            raise ValueError("best_of must be 1 when using greedy sampling."
                             f"Got {self.best_of}.")
        if self.top_p < 1.0 - _SAMPLING_EPS:
            raise ValueError("top_p must be 1 when using greedy sampling.")
        if self.top_k != -1:
            raise ValueError("top_k must be -1 when using greedy sampling.")

    def __repr__(self) -> str:
        return (f"SamplingParams(n={self.n}, "
                f"best_of={self.best_of}, "
                f"presence_penalty={self.presence_penalty}, "
                f"frequency_penalty={self.frequency_penalty}, "
                f"temperature={self.temperature}, "
                f"top_p={self.top_p}, "
                f"top_k={self.top_k}, "
                f"use_beam_search={self.use_beam_search}, "
                f"stop={self.stop}, "
                f"ignore_eos={self.ignore_eos}, "
                f"max_tokens={self.max_tokens}, "
                f"logprobs={self.logprobs})")


class LogitsWarper(ABC):
    def __init__(self, logit_scale: float = -100, logit_bias: float = -100, **kwargs):
        self.logit_scale = logit_scale
        self.logit_bias = logit_bias

    def __call__(self, logit: float) -> float:
        return logit * self.logit_scale + self.logit_bias


class Selection(LogitsWarper):
    """Logit warper that selects logits from a given list."""

    def __init__(self, options: List[str], options_tokens: List[int], **kwargs) -> None:
        super().__init__()
        self.options = options
        self.options_tokens = options_tokens
        # seq_id is a universal counter in llm_engine
        # how to maintain options per sequence?
        # This is one instance per sequence group, so all taken care of
        self.sequence_state = {}
        # for option in self.options:
        #     tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(option))
        #     self.options.append(tokens)
        # We need to search through the options space in a Trie based fashion
        # will help with branching and expressiveness
        # re.compile and looping over 50k tokens is not feasible
        # Algorithm is:
        #  - Generate a Trie with all possible tokenisation combinations
        #  - Each branch will have a different version of an option
        #  - Get the current step number from the state
        #  - Bias all tokens except the tokens of interest to zero
        #  - Sampler will sample from the logits
        #  - Get the output token and update the state and step number within the Trie
        #  - Short circuting can be done if we know the current branch doesn't branch out in later steps
        #  - If we see it does, go step by step
        #  - Can we do sampling in GPU? Can we maintain this state and have Tries in GPU?
        # We can't do this for Regex output control

    def __call__(self, seq_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        max_prob = 0
        seq_id = seq_ids[0]  # For now
        # TODO: look into tensor shape of prob
        # Edge cases:
        #   - if only one option, return that
        #   - if more than one option has the same starting token, need to branch to the right one
        #   - if we figure out no other option is possible, stop generation and return the best option tokens
        #   - check for end here or in stopping code?

        print("--> ", logits.shape)
        if not self.sequence_state[seq_id]:
            for i, option in enumerate(self.options_tokens):
                if logits[option[0]] > max_prob:
                    max_prob = logits[option[0]]
                    self.sequence_state[seq_id] = [i, option[0]]
                    logits[option[0]] = 1000  # torch.inf
            return logits
        else:
            # store option number and last token id
            curr_option = self.options[self.sequence_state[seq_id][0]]
            last_token = self.sequence_state[seq_id][1]

            # check for max len here
            new_token = curr_option[curr_option.index(last_token) + 1]
            logits[new_token] = 1000  # torch.inf
            self.sequence_state[seq_id][1] = new_token
            return logits

        # self.sequence_state.append(logit)


# class SequenceStopper():
#     def __init__(self, stop_sequence) -> None:
#         super().__init__()
