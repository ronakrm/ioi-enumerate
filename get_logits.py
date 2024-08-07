from transformer_lens.utils import test_prompt
from transformer_lens import HookedTransformer

from ioi_datasets import IOIFullDataset, ioi_eval_sliced
from ioi_constants import BABA_TEMPLATES, NOUNS_DICT

from names import get_names

import pandas as pd

DEVICE = 'cuda'
BATCH_SIZE = 1024
RES_FILE = "ioi_logit_dump.csv"

# load tokenizer, model, data
model = HookedTransformer.from_pretrained("gpt2-small").to(DEVICE)
tokenizer = model.tokenizer
names = get_names('original_ioi')

my_ds = IOIFullDataset(
  tokenizer,
  templates=BABA_TEMPLATES,
  names=names,
  nouns=NOUNS_DICT,
  max_gen_len=100000000,
  prepend_bos=True,
)
print('Total dataset size:', len(my_ds))

# run forward passes
res = ioi_eval_sliced(model, dataset=my_ds, res_file=RES_FILE, batch_size=BATCH_SIZE)
indiv_df = pd.DataFrame(res)
indiv_df.to_csv(RES_FILE, index=False)

bad_res = []
for tmp in res:
  if tmp['logit_diff'] < 0:
    bad_res.append(tmp)


print(f"number of incorrect diffs: {len(bad_res)}")
test_prompt(tokenizer.decode(tokenizer.encode(bad_res[0]['text'])), '_', model)
