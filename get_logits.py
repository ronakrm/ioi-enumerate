import argparse
from transformer_lens.utils import test_prompt
from transformer_lens import HookedTransformer

from ioi_datasets import IOIFullDataset, ioi_eval_sliced
from ioi_constants import BABA_TEMPLATES, NOUNS_DICT

from names import get_names

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--res_file', default='tmp.csv', type=str)
parser.add_argument('--model_name', default='gpt2-small', type=str)
parser.add_argument('--max_num_samples', default=1000, type=int)
args = parser.parse_args()

# load tokenizer, model, data
model = HookedTransformer.from_pretrained(args.model_name).to(args.device)
tokenizer = model.tokenizer
names = get_names('original_ioi')

my_ds = IOIFullDataset(
  tokenizer,
  templates=BABA_TEMPLATES,
  names=names,
  nouns=NOUNS_DICT,
  max_gen_len=args.max_num_samples,
  prepend_bos=True,
)
print('Total dataset size:', len(my_ds))

# run forward passes
res = ioi_eval_sliced(model, dataset=my_ds, res_file=args.res_file, batch_size=args.batch_size)
indiv_df = pd.DataFrame(res)
indiv_df.to_csv(args.res_file, index=False)

bad_res = []
for tmp in res:
  if tmp['logit_diff'] < 0:
    bad_res.append(tmp)


print(f"number of incorrect diffs: {len(bad_res)}")
test_prompt(tokenizer.decode(tokenizer.encode(bad_res[0]['text'])), '_', model)
