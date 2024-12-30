import os
from pathlib import Path

from transformers import OPTForCausalLM
import torch
import torch.nn as nn



class GPTPromptTuningMixin:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        soft_prompt_path: str = None,
        n_tokens: int = 100,
        initialize_from_vocab: bool = True,
        random_range: float = 0.5,
        random_selection: bool = False,
        **kwargs,
    ):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Make sure to freeze Tranformers model
        for param in model.parameters():
            param.requires_grad = False

        if soft_prompt_path is not None:
            model.set_soft_prompt_embeds(soft_prompt_path)
        elif n_tokens is not None:
            print("Initializing soft prompt...")
            print(model)
            model.initialize_soft_prompt(
                n_tokens=n_tokens,
                initialize_from_vocab=initialize_from_vocab,
                random_range=random_range,
                random_selection=random_selection
            )

        return model


    def set_soft_prompt_embeds(
        self,
        soft_prompt_path: str,
    ) -> None:
        """
        Args:
            soft_prompt_path: torch soft prompt file path

        """
        self.soft_prompt = torch.load(
            soft_prompt_path, map_location=torch.device("cpu")
        )
        self.n_tokens = self.soft_prompt.num_embeddings
        print(f"Set soft prompt! (n_tokens: {self.n_tokens})")


    def initialize_soft_prompt(
        self,
        n_tokens: int = 20,
        initialize_from_vocab: bool = True,
        random_range: float = 0.5,
        random_selection: bool = False
    ) -> None:
        self.n_tokens = n_tokens
        if initialize_from_vocab:
            init_prompt_value = self.model.decoder.embed_tokens.weight[:n_tokens].clone().detach()
        elif random_selection:
            total_tokens = len(self.model.decoder.embed_tokens.weight)
            selected_indices = torch.randperm(total_tokens)[:n_tokens]
            init_prompt_value = self.model.decoder.embed_tokens.weight[selected_indices].clone().detach()
        else:
            dtype = self.model.decoder.embed_tokens.weight.dtype
            device = self.model.decoder.embed_tokens.weight.device
            init_prompt_value = torch.FloatTensor(n_tokens, self.config.word_embed_proj_dim, device=device).uniform_(
                -random_range, random_range
            ).to(dtype=dtype)
        self.soft_prompt = nn.Embedding(n_tokens, self.config.word_embed_proj_dim)
        # Initialize weight
        self.soft_prompt.weight = nn.parameter.Parameter(init_prompt_value)


    def _cat_learned_embedding_to_input(self, input_ids) -> torch.Tensor:
        bsize, length = input_ids.shape
        ntoken, dim = self.soft_prompt.weight.shape
        device = self.soft_prompt.weight.device
        dtype = self.soft_prompt.weight.dtype

        inputs_embeds = self.model.decoder.embed_tokens(input_ids.to(device))

        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        output_embeds = torch.empty(bsize, length + ntoken, dim, device=device, dtype=dtype)
        for i in range(bsize):
            non_pad_position = torch.nonzero(input_ids[i] != self.config.pad_token_id)[0][0]
            output_embeds[i, :non_pad_position] = inputs_embeds[i, :non_pad_position]
            output_embeds[i, non_pad_position:non_pad_position + ntoken] = self.soft_prompt.weight.clone()
            output_embeds[i, non_pad_position + ntoken:] = inputs_embeds[i, non_pad_position:]

        return output_embeds


    def _extend_labels(self, labels, ignore_index=-100) -> torch.Tensor:
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)

        n_batches = labels.shape[0]
        return torch.cat(
            [
                torch.full((n_batches, self.n_tokens), ignore_index).to(labels.device),
                labels,
            ],
            dim=1,
        )


    def _extend_attention_mask(self, attention_mask):
        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        # padding side is left, so we append 1 to the right
        return torch.cat([attention_mask, torch.full((n_batches, self.n_tokens), 1, device=attention_mask.device)], dim=1).to(self.model.device)


    def save_soft_prompt(self, path: str, filename: str = "soft_prompt.model"):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.soft_prompt, os.path.join(path, filename))
        # print(f"Saved soft prompt: {os.path.join(path, filename)}")


    def save_model(self, path: str):
        self.model.save_pretrained(path)


    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if input_ids is not None:
            inputs_embeds = self._cat_learned_embedding_to_input(input_ids).to(
                self.device
            )

        if labels is not None:
            labels = self._extend_labels(labels).to(self.device)

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask).to(self.device)

        # Drop most of the args for now
        return super().forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            return_dict=return_dict,
        )


class OPTPromptTuningLM(GPTPromptTuningMixin, OPTForCausalLM):
    def __init__(self, config):
        super().__init__(config)