# test_model_load.py
from config.model_config import load_llama_model
tokenizer, model = load_llama_model()
print("Loaded model and tokenizer successfully.")
