from typing import Union

from torch.utils.data import Dataset
from tqdm import tqdm


def preprocess_data(
    data, input_template=None, input_key="input", apply_chat_template=None, image_text_to_text=False
) -> Union[str, dict]:
    if image_text_to_text:
        try:
            input_key, image_key = input_key.split(",")
        except ValueError:
            raise ValueError("input_key must contain both text and image keys, separated by a comma.")
        if apply_chat_template:
            chat = data[input_key]
            if isinstance(chat, str):
                chat = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": chat}]}]
            prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        else:
            prompt = data[input_key]
            if input_template:
                prompt = input_template.format(prompt)
        image = data[image_key]
        return {"text": prompt, "image": image}
    else:
        if apply_chat_template:
            chat = data[input_key]
            if isinstance(chat, str):
                chat = [{"role": "user", "content": chat}]
            prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        else:
            prompt = data[input_key]
            if input_template:
                prompt = input_template.format(prompt)
        return prompt


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.image_text_to_text = getattr(self.strategy.args, "image_text_to_text", False)

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(
                data, input_template, input_key, apply_chat_template, self.image_text_to_text
            )
            self.prompts.append(prompt)

        if self.image_text_to_text:
            self.collate_fn = lambda x: x
        else:
            self.collate_fn = None

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]
