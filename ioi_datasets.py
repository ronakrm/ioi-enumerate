import random
from typing import Dict, List, Optional
from transformer_lens.utils import test_prompt

from eindex import eindex
import einops
import torch
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


class IOIDataset(Dataset):
    """
    Dataset for Indirect Object Identification tasks.
    Paper: https://arxiv.org/pdf/2211.00593.pdf

    Example:

    .. code-block:: python

        >>> from transformer_lens.evals import ioi_eval, IOIDataset
        >>> from transformer_lens.HookedTransformer import HookedTransformer

        >>> model = HookedTransformer.from_pretrained('gpt2-small')
        Loaded pretrained model gpt2-small into HookedTransformer

        >>> # Evaluate like this, printing the logit difference
        >>> print(round(ioi_eval(model, num_samples=100)["Logit Difference"], 3))
        5.476

        >>> # Can use custom dataset
        >>> ds = IOIDataset(
        ...     tokenizer=model.tokenizer,
        ...     num_samples=100,
        ...     templates=['[A] met with [B]. [B] gave the [OBJECT] to [A]'],
        ...     names=['Alice', 'Bob', 'Charlie'],
        ...     nouns={'OBJECT': ['ball', 'book']},
        ... )
        >>> print(round(ioi_eval(model, dataset=ds)["Logit Difference"], 3))
        5.397
    """

    def __init__(
        self,
        tokenizer,
        templates: Optional[List[str]] = None,
        names: Optional[List[str]] = None,
        nouns: Optional[Dict[str, List[str]]] = None,
        num_samples: int = 1000,
        symmetric: bool = False,
        prepend_bos: bool = True,
    ):

        random.seed(3245)

        self.tokenizer = tokenizer
        self.prepend_bos = prepend_bos

        self.templates = templates if templates is not None else self.get_default_templates()
        self.names = names if names is not None else self.get_default_names()
        self.nouns = nouns if nouns is not None else self.get_default_nouns()

        self.samples = []
        for _ in range(num_samples // 2 if symmetric else num_samples):
            # If symmetric, get_sample will return two samples
            self.samples.extend(self.get_sample(symmetric=symmetric))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = self.tokenizer.encode(sample["text"])
        if self.prepend_bos:
            prompt = [self.tokenizer.bos_token_id] + prompt

        return {
            "prompt": torch.LongTensor(prompt),
            "IO": torch.LongTensor(self.tokenizer.encode(sample["IO"])),
            "S": torch.LongTensor(self.tokenizer.encode(sample["S"])),
        }

    def get_sample(self, symmetric=False) -> List[Dict[str, str]]:
        template: str = random.choice(self.templates)
        for noun_type, noun_list in self.nouns.items():
            template = template.replace(f"[{noun_type}]", random.choice(noun_list))

        samples: List[Dict[str, str]] = []

        # Sample two names without replacement
        names = random.sample(self.names, 2)
        sample = template.replace("[A]", names[0])
        sample = sample.replace("[B]", names[1])
        # Prepend spaces to IO and S so that the target is e.g. " Mary" and not "Mary"
        samples.append({"text": sample, "IO": " " + names[0], "S": " " + names[1]})

        if symmetric:
            sample_2 = template.replace("[A]", names[1])
            sample_2 = sample_2.replace("[B]", names[0])
            samples.append({"text": sample_2, "IO": " " + names[1], "S": " " + names[0]})

        return samples

    @staticmethod
    def get_default_names():
        return ["John", "Mary"]

    @staticmethod
    def get_default_templates():
        return [
            "[A] and [B] went to the [LOCATION] to buy [OBJECT]. [B] handed the [OBJECT] to [A]",
            "Then, [B] and [A] went to the [LOCATION]. [B] gave the [OBJECT] to [A]",
        ]

    @staticmethod
    def get_default_nouns():
        return {
            "LOCATION": ["store", "market"],
            "OBJECT": ["milk", "eggs", "bread"],
        }

class IOIFullDataset(Dataset):
    """
    Copy of IOIDataset but with the ability to return all metadata we want to track.
    """

    def __init__(
        self,
        tokenizer,
        templates: Optional[List[str]] = None,
        names: Optional[List[str]] = None,
        nouns: Optional[Dict[str, List[str]]] = None,
        prepend_bos: bool = True,
        max_gen_len: int = 128,
        seed: int = 42,
    ):
        random.seed(seed)
        self.seed = seed
        self.tokenizer = tokenizer
        self.prepend_bos = prepend_bos

        self.templates = (
            templates if templates is not None else self.get_default_templates()
        )
        self.names = names if names is not None else self.get_default_names()
        self.nouns = nouns if nouns is not None else self.get_default_nouns()

        self.samples = []

        print(len(names))
        print(len(nouns["PLACE"]))
        print(len(nouns["OBJECT"]))
        print(len(templates))

        full_size = len(names)*(len(names)-1)*len(nouns["PLACE"])*len(nouns["OBJECT"])*len(templates)
        total_samples = min(max_gen_len, full_size)
        pbar = tqdm(total = total_samples)

        for template in templates:
            # hard coded for now for 2 noun types
            for place in nouns[list(nouns.keys())[0]]:
                for object in nouns[list(nouns.keys())[1]]:
                    for s in names:
                        for io in names:
                            if s == io:
                                continue

                            meta_dict = {
                                "template": template,
                                "PLACE": place,
                                "OBJECT": object,
                                "S": s,
                                "IO": io,
                            }

                            built_sample = self.get_sample(meta_dict)
                            self.samples.append(built_sample)
                            pbar.update(1)

                            if len(self.samples) >= max_gen_len:
                                return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = self.tokenizer.encode(sample["text"])
        if self.prepend_bos:
            prompt = [self.tokenizer.bos_token_id] + prompt

        return {
            "prompt": torch.LongTensor(prompt),
            "IO": torch.LongTensor(self.tokenizer.encode(" " + sample["IO"])),
            "S": torch.LongTensor(self.tokenizer.encode(" " + sample["S"])),
            "extras": sample,
        }

    def get_sample(self, meta_dict) -> List[Dict[str, str]]:

        sample_dict = meta_dict
        text = meta_dict["template"]
        text = text.replace(f"[PLACE]", meta_dict["PLACE"])
        text = text.replace(f"[OBJECT]", meta_dict["OBJECT"])

        text = text[:-4]
        sample = text.replace("[A]", meta_dict["IO"])
        sample = sample.replace("[B]", meta_dict["S"])
        sample_dict["text"] = sample

        return sample_dict


@torch.inference_mode()
def ioi_eval_sliced(
    model, dataset, batch_size=8, tokenizer=None, symmetric=False, device='cpu'
):
    """
    Copy of existing ioi_eval but with the ability to return all logit differences
    with metadata dictionary.
    """
    if tokenizer is None:
        tokenizer = model.tokenizer

    def collate(samples):
        prompts = [sample["prompt"] for sample in samples]
        # padded_prompts = torch.nn.utils.rnn.pad_sequence(prompts, batch_first=True)
        padded_prompts = torch.stack(prompts)
        return {
            "prompt": padded_prompts,
            "IO": [sample["IO"][0] for sample in samples],
            "S": [sample["S"][0] for sample in samples],
            "prompt_length": [p.shape[0] for p in prompts],
            "extras": [sample["extras"] for sample in samples],
        }

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate
    )

    res_list = []
    for batch in tqdm(data_loader):
        prompt = batch["prompt"].to(device)
        
        io, s = torch.stack(batch["IO"]), torch.stack(batch["S"])
        batch_logits = model(prompt, return_type="logits")[:, -1, :].detach().cpu()
        logit_diffs = eindex(batch_logits, io, 'b [b]') \
                     - eindex(batch_logits, s, 'b [b]')

        for i in range(len(batch['prompt'])):
            res_dict = batch["extras"][i]
            res_dict["logit_diff"] = logit_diffs[i].item()
            res_list.append(res_dict)

    return res_list


def get_pred(prompt, io, s):
  logits = model(prompt.to('cuda'), return_type="logits")[0, -1, :]
  max_logit = logits.argmax().item()
  logit_diff_io_minus_s = logits[io].item() - logits[s].item()
  return max_logit, logit_diff_io_minus_s

def get_incorrect_res(prompt_list):
  incorrect_res = []
  incorrect_order = []
  for i in tqdm(range(len(prompt_list))):
    data = prompt_list[i]
    prompt = data['prompt']
    pred_token, logit_diff_io_minus_s = get_pred(prompt[:-1], data['IO'][0], data['S'][0])
    if pred_token != data['IO'][0].item():
      incorrect_res.append(prompt)
    if logit_diff_io_minus_s < 0:
      incorrect_order.append(prompt)
  return incorrect_res, incorrect_order
