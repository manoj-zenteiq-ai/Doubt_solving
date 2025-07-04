# config/model_config.py

from transformers import AutoTokenizer, AutoModelForCausalLM

def load_tinyllama_model(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", load_in_8bit=True):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        load_in_8bit=load_in_8bit
    )
    return tokenizer, model
