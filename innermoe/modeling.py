from transformers import LlamaConfig, LlamaModel
import torch.nn as nn
import torch
from typing import Optional
import copy
from peft import get_peft_model
import innermoe.config as config


class InnerMoERM(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.base_config = LlamaConfig.from_pretrained(config.model_name_or_path)  # Llamaconfig works well on Qwen models
        self.base_model = LlamaModel(self.base_config)
        
        self.base_value_head = nn.Linear(self.base_model.config.hidden_size, 1)
        self.lora_config = config.lora_config
        self.lora_value_heads = {}  # it will be initialized in Phase 2
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.phase = 1
        self.device = config.device


    def change_phase(self, phase):
        self.phase = phase
        if phase == 3:  # for Phase 3, add an MLP to aggregate all LoRA experts' outputs
            self.final_value_head1 = nn.Linear(self.base_config.hidden_size * len(self.lora_value_heads.keys()), \
                self.base_config.hidden_size).to(self.device)
            self.final_value_head2 = nn.Linear(self.base_config.hidden_size, 1).to(self.device)


    def change_cap(self, cap):
        if cap not in self.lora_value_heads:  # if the corresonding LoRA expert is not created, create it by injecting LoRA layers
            self.base_model = get_peft_model(self.base_model, self.lora_config, adapter_name=cap)
            self.lora_value_heads[cap] = copy.deepcopy(self.base_value_head)  # each LoRA expert has its own value head learning from base model
        self.now_cap = cap
        self.base_model.set_adapter(cap)  # activate the corresonding LoRA expert


    def forward(self,
                input_ids: torch.LongTensor, 
                attention_mask: Optional[torch.Tensor] = None,
                **kargs) -> torch.Tensor:
        def get_hidden_states():  # this function is used to get the last hidden states of an LM
            outputs = self.base_model(input_ids,attention_mask=attention_mask, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]
            if attention_mask is None:
                last_hidden_states = last_hidden_states[:, -1]
            else:
                last_index = attention_mask.cumsum(dim=1).argmax(dim=1)
                last_hidden_states = last_hidden_states.gather(1, last_index.view(-1, 1, 1).expand(-1, 1, last_hidden_states.size(-1))).squeeze(1)
            return last_hidden_states

        if self.phase == 1:  # for Phase 1, use the base model to predict value
            last_hidden_states = get_hidden_states()
            value = self.base_value_head(last_hidden_states).squeeze(-1)
        elif self.phase == 2:  # for Phase 2, use the current-activated LoRA experts to predict value
            last_hidden_states = get_hidden_states()
            value = self.lora_value_heads[self.now_cap](last_hidden_states).squeeze(-1)
        else:  # for Phase 3, first obtain all Lora experts' outputs, then aggregate them with the MLP
            with torch.no_grad():  # in Phase 3, the base model and all LoRA experts are frozen
                last_hidden_states_list = []
                for cap in self.lora_value_heads.keys():
                    self.base_model.set_adapter(cap)
                    last_hidden_states = get_hidden_states()
                    last_hidden_states_list.append(last_hidden_states)
                final_last_hidden_states = torch.cat(last_hidden_states_list, dim=-1)
            h = self.final_value_head1(final_last_hidden_states)  # only train the final MLP
            h = self.prelu(h)
            value = self.final_value_head2(h).squeeze(-1)
        return value  # return the final one-dimensional reward value


    def get_reward(self, context, tokenizer):
        self.eval()
        with torch.no_grad():
            user = '<extra_0>'
            bot = '<extra_1>'
            end = '<extra_2>'
            prompt = f"{user}{context['src'][-1]}{bot}{context['tgt'][-1]}{end}"
            tokenized = tokenizer(prompt, return_tensors="pt").to(config.device)
            reward = self.forward(tokenized['input_ids'], tokenized['attention_mask'])
        return reward.item()
