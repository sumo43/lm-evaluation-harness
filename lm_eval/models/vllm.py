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
from lm_eval import utils
from lm_eval.utils import MultiTokenEOSCriteria, stop_sequences_criteria

TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]

def get_result(response: dict, ctxlen: int) -> Tuple[float, bool]:
    """Process results from OpenAI API response.

    :param response: dict
        OpenAI API Response
    :param ctxlen: int
        Length of context (so we can slice them away and only keep the predictions)
    :return:
        continuation_logprobs: np.array
            Log probabilities of continuation tokens
        is_greedy: bool
            whether argmax matches given continuation exactly
    """
    is_greedy = True
    logprobs = list(response.prompt_logprobs[1].values())
    for i in range(ctxlen, len(logprobs)):

        if len(logprobs[i]) == 11:
            is_greedy = False
            logprobs[i] = logprobs[i][1:]

    continuation_logprobs = sum(logprobs[ctxlen:])

    return continuation_logprobs, is_greedy

@register_model("vllm")
class VLLM(LM):
    AUTO_CONFIG_CLASS: transformers.AutoConfig = transformers.AutoConfig
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
    AUTO_TOKENIZER_CLASS: transformers.AutoTokenizer = transformers.AutoTokenizer
    _DEFAULT_MAX_LENGTH: int = 2048
    REQ_CHUNK_SIZE = 1

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

        self._rank = 1
        self._world_size = 1

        self.llm = LLM(model=pretrained)
        self.tokenizer = self._create_auto_tokenizer(
            pretrained=pretrained,
            revision=revision,
            subfolder=subfolder,
            tokenizer=pretrained,
        )

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
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                # end of text as context
                context_enc, continuation_enc = [self.eot_token_id], self.tok_encode(
                    continuation
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def _loglikelihood_tokens(self, requests):
        res = []

        def _collate(x):
            # this doesn't efficiently handle last-token differences yet, but those are kinda annoying because
            # it's not guaranteed that the 100 or so logprobs we get to see actually contain all the continuations
            # we care about, and so we need some kind of backup for when it isn't
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)
        
        re_ord = utils.Reorderer(requests, _collate)

        for chunk in tqdm(
            list(utils.chunks(re_ord.get_reordered(), self.REQ_CHUNK_SIZE))
        ):

            inps = []
            ctxlens = []
            for cache_key, context_enc, continuation_enc in chunk:
                # max_length+1 because the API takes up to 2049 tokens, including the first context token
                inp = (context_enc + continuation_enc)[-(self.max_length + 1) :]
                # TODO: the logic is much simpler if we just look at the length of continuation tokens
                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - (self.max_length + 1)
                )

                inps.append(inp)
                ctxlens.append(ctxlen)

            sampling_params = SamplingParams(max_tokens=1, 
                                         temperature=0, 
                                         top_p=1,
                                         prompt_logprobs=10)
            outputs = self.llm.generate(prompt_token_ids=inps, 
                                    sampling_params=sampling_params)
            
            for resp, ctxlen, (cache_key, context_enc, continuation_enc) in zip(
                outputs, ctxlens, chunk
            ):
                answer = get_result(resp, ctxlen)

                res.append(answer)

                # partial caching
                #if cache_key is not None:
                #    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

        return re_ord.get_original(res)

    def generate_until(self, requests):
        raise NotImplementedError # need this 

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError # need this

    def _model_call(self, inps):
        raise NotImplementedError # dont need this

    def generate_until(self, requests):
        raise NotImplementedError
    
    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None):
        """ """
        if add_special_tokens is None:
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                add_special_tokens = False
            elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
                add_special_tokens = True

        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding
    
    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tok_encode(context + continuation, add_special_tokens=False)
        context_enc = self.tok_encode(context, add_special_tokens=False)

        # whole_enc = self.tok_encode(context + continuation)
        # context_enc = self.tok_encode(context, add_special_tokens=False)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc