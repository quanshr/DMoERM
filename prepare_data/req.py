from transformers import AutoTokenizer, AutoModelForCausalLM
import prepare_data.config as config


class LLM():
    """
    For cost and efficiency considerations, we provide an implementation using a local LLM, which is different from our original implementation (calling Ernie 4.0 API).
    """

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            device_map="auto",
            trust_remote_code=True
        ).to(config.device)
        self.llm.eval()


    def gen_res(self, prompt):
        response, _ = self.llm.chat(self.tokenizer, prompt, history=None)
        return response
