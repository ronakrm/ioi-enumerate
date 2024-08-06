from transformer_lens import HookedTransformer
from transformer_lens.utils import test_prompt


DEVICE = 'cpu'

# load tokenizer, model, data
model = HookedTransformer.from_pretrained("gpt2-small").to(DEVICE)
tokenizer = model.tokenizer

# YOU NEED THESE SPACES
subject = " Michael"
target = " Laura"
prompt = "<|endoftext|>Then, Michael and Laura went to the store. Michael gave a ring to"

subj_tok, io_tok = tokenizer.encode(subject), tokenizer.encode(target)

logits = model(prompt, return_type="logits")[0, -1, :].detach().cpu()
subj_logit, io_logit = logits[subj_tok], logits[io_tok]

print(f"logits from vanilla forward: subj {subj_logit} io {io_logit}")

test_prompt(prompt, target, model)
