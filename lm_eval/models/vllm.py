import random
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from typing import List, Mapping, NewType, Optional, Tuple, Union
from vllm import LLM, SamplingParams
import torch
from transformers import BatchEncoding
import transformers
from tqdm import tqdm
from itertools import groupby

TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]

@register_model("vllm")
class VLLM(LM):
    AUTO_CONFIG_CLASS: transformers.AutoConfig = transformers.AutoConfig
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
    AUTO_TOKENIZER_CLASS: transformers.AutoTokenizer = transformers.AutoTokenizer
    _DEFAULT_MAX_LENGTH: int = 2048

    def __init__(
        self,
        pretrained: str,
        subfolder: Optional[str] = None,
        revision: Optional[str] = "main",
        add_special_tokens: Optional[bool] = None,
        batch_size: Optional[int] = 1,
        max_gen_toks: Optional[int] = 1024,
        max_length: Optional[int] = None,
        trust_remote_code: Optional[bool] = False,
        tensor_parallel_size: Optional[int] = 1,
        dtype: Optional[Union[str, torch.dtype]] = 'bfloat16'
    ) -> None:

        self._max_gen_toks = max_gen_toks
        self._max_length = max_length
        self._batch_size = batch_size
        self._trust_remote_code = trust_remote_code
        self._add_special_tokens = add_special_tokens
        self._config = self.AUTO_CONFIG_CLASS.from_pretrained(
            pretrained,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
            trust_remote_code=self._trust_remote_code,
        )
        self.llm = LLM(model=pretrained, 
                       tensor_parallel_size=tensor_parallel_size,
                       dtype=dtype,
                       swap_space=64)
        self.tokenizer = self._create_auto_tokenizer(
            pretrained=pretrained,
            revision=revision,
            subfolder=subfolder,
            tokenizer=pretrained,
        )

    #@classmethod
    #def create_from_arg_string(cls, arg_string, additional_config=None):
    #    return cls()
            
    def _create_auto_tokenizer(
        self,
        *,
        pretrained: str,
        revision: str,
        subfolder: str,
        tokenizer: Optional[str] = None,
    ) -> transformers.PreTrainedTokenizer:
        """Returns a pre-trained tokenizer from a pre-trained tokenizer configuration."""
        tokenizer = self.AUTO_TOKENIZER_CLASS.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
            trust_remote_code=self._trust_remote_code,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer
    
    def _model_generate(
        self,
        inputs: transformers.BatchEncoding,
        max_tokens: int,
        stop: Optional[List[str]] = None,
        num_return_sequences: int = 1,
        num_return_sequences_batch: int = -1, # Doesn't do anything. Just here to match the signature of the other models.
        temperature: float = 0.0, 
        top_p: float = 1,
    ) -> TokenSequence:

        if isinstance(stop, str):
            stop = [stop]

        input_ids = inputs["input_ids"][:, self.max_gen_toks-self.max_length:]

        # Decode each back to a string
        contexts = self.tok_decode(input_ids)

        bsz = len(input_ids)

        output_texts = []
        sampling_params = SamplingParams(max_tokens=max_tokens, 
                                         temperature=temperature, 
                                         top_p=top_p,
                                         stop=stop, 
                                         n=num_return_sequences)
        outputs = self.llm.generate(prompts=contexts, 
                                    sampling_params=sampling_params,
                                    use_tqdm=bsz > 1)
        
        # Sort by request_id
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        for output in outputs:
            generations = [gen.text for gen in output.outputs]
            if len(generations) == 1:
                generations = generations[0]
            output_texts.append(generations)

        return output_texts

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model.
        NOTE: Different model configurations have different max sequence length
        attribute names.
            - n_positions: (CTRLConfig)
            - max_position_embeddings: (BartConfig, RoFormerConfig)
            - n_ctx: (GPT2Config)
        NOTE: For relative position encoded models you should specify the max
        sequence length of the model in the constructor via `max_length`.
        """
        if self._max_length is not None:
            return self._max_length
        # Try to get the sequence length from the model config.
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self._config, attr):
                return getattr(self._config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH


    def tok_encode(self, string: str) -> TokenSequence:
        # TODO: Merge `tok_encode_batch` here.
        return self.tokenizer.encode(string, add_special_tokens=self.add_special_tokens)

    def tok_batch_encode(self, strings: List[str]) -> TokenSequence:
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )
    
    def tok_decode(self, tokens: torch.LongTensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def loglikelihood(self, requests):
        return NotImplementedError

    def generate_until(self, requests):
        return NotImplementedError # need this 

    def loglikelihood_rolling(self, requests):
        return NotImplementedError # need this

    def _model_call(self, inps):
        raise NotImplementedError # dont need this

    def generate_until(self, requests):
        raise NotImplementedError