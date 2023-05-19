import torch
import transformers
from typing import Optional
from lm_eval.base import BaseLM
from open_lm.model import  _MODEL_CONFIGS, Transformer, ModelArgs
from training.file_utils import pt_load
from tiktoken import get_encoding
from training.generation import Generator
from copy import deepcopy

def modify_tokens(tokens):
    for token in tokens:
        if token in [50281, 50282]:
            yield 50256
        else:
            yield token

class OpenLM(BaseLM):
    def __init__(
        self,
        path_to_checkpoint,
        device="cuda",
        pretrained_model="gpt2",
        revision="main",
        low_cpu_mem_usage=None,
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        load_in_8bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = True,
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained_model, str)
        assert isinstance(batch_size, (int,str))

        device_list = set(["cuda", "cpu"] + [f'cuda:{i}' for i in range(torch.cuda.device_count())])
        if device and device in device_list:
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # TODO: update this to be less of a hack once subfolder is fixed in HF
        revision = revision + ("/" + subfolder if subfolder is not None else "")

        cfg =  deepcopy(_MODEL_CONFIGS[pretrained_model])

        model_args = ModelArgs(
            dim=cfg['hidden_dim'],
            n_layers=cfg['n_layers'],
            n_heads=cfg['n_heads'],
            seq_len=cfg['seq_len'],
            vocab_size=cfg['vocab_size'],
            pre_ln=cfg['pre_ln'],
            pos_embed_type=cfg['pos_embed_type'],
            weight_tying=cfg['weight_tying'],
            attn_type=cfg['attn_type'],
        )
        model = Transformer(model_args)

        
        self.gpt2 = model.to(self.device)
        checkpoint = pt_load(path_to_checkpoint, map_location='cpu')
        sd = checkpoint['state_dict']
        #sd = {k[len('module.'):]: v for k, v in sd.items()}
        
        self.gpt2.load_state_dict(sd)
        self.gpt2.eval()

        self.tokenizer = get_encoding("p50k_base")

        self.vocab_size = 50304
        self.generator = Generator(model)

        # setup for automatic batch size detection
        if batch_size == 'auto': 
            self.batch_size_per_gpu = batch_size
        else:
            self.batch_size_per_gpu = int(batch_size) 

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return 50256

    @property
    def max_length(self):
        return 2048

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens):
        tokens = modify_tokens(tokens)
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.gpt2(inps)

    def _model_generate(self, context, max_length, eos_token_id):
        generation_kwargs = {'temperature': 0.0, 'max_gen_len': max_length}
        return self.generator.generate(context)

