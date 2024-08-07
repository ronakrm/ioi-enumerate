import random
from typing import Dict, List, Optional

from eindex import eindex
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


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

        print(len(names))
        print(len(nouns["PLACE"]))
        print(len(nouns["OBJECT"]))
        print(len(templates))

        full_size = len(names)*(len(names)-1)*len(nouns["PLACE"])*len(nouns["OBJECT"])*len(templates)
        total_samples = min(max_gen_len, full_size)
        pbar = tqdm(total = total_samples)

        self.samples = []
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
                            if len(built_sample['prompt_toks']) != 16:
                                continue

                            self.samples.append(built_sample)
                            pbar.update(1)

                            if len(self.samples) >= max_gen_len:
                                return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # prompt = self.tokenizer.encode(sample["text"])
        prompt = sample['prompt_toks']
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

        print(sample_dict['text'])
        sample_dict['prompt_toks'] = self.tokenizer.encode(sample_dict["text"])
        return sample_dict


@torch.inference_mode()
def ioi_eval_sliced(
    model, dataset, res_file, batch_size=8, tokenizer=None, symmetric=False, device='cpu'
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
        batch_logits = model(prompt, return_type="logits")[:, -1, :]
        logit_diffs = eindex(batch_logits, io, 'b [b]') \
                     - eindex(batch_logits, s, 'b [b]')

        for i in range(len(batch['prompt'])):
            res_dict = batch["extras"][i]
            res_dict["logit_diff"] = logit_diffs[i].detach().cpu().item()
            res_list.append(res_dict)

    return res_list

