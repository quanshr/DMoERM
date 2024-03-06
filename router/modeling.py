from transformers import LlamaConfig, LlamaModel
import torch.nn as nn
import torch
from typing import Optional
import router.config as config


class Router(nn.Module):

    def __init__(self):
        super().__init__()
        model_config = LlamaConfig.from_pretrained(config.model_name_or_path)  # Llamaconfig works well on Qwen models
        self.model = LlamaModel(model_config)
        self.value_head = nn.Linear(model_config.hidden_size, len(config.cat_list))
        self.softmax = nn.Softmax(dim=1)
    
    
    def forward(self,
                input_ids: torch.LongTensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(input_ids,attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        if attention_mask is None:
            last_hidden_states = last_hidden_states[:, -1]
        else:
            last_index = attention_mask.cumsum(dim=1).argmax(dim=1)
            last_hidden_states = last_hidden_states.gather(1, last_index.view(-1, 1, 1).expand(-1, 1, last_hidden_states.size(-1))).squeeze(1)
        prob = self.value_head(last_hidden_states)
        prob = self.softmax(prob).squeeze()
        return prob


    def route(self, context, tokenizer):
        user_token = '<extra_0>'
        self.eval()
        with torch.no_grad():
            src = ''
            for query in context['src']:
                src += user_token + query
                
            tokenized = tokenizer(src, return_tensors="pt").to(config.device)
            
            output = self(tokenized['input_ids'], tokenized['attention_mask'])
            _, predicted = torch.max(output, 0)
        
        return predicted.item()
