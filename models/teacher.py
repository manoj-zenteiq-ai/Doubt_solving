from utils.io_utils import load_prompt, format_prompt
from transformers import PreTrainedModel, PreTrainedTokenizer

def teach_concept(concept: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> str:
    prompt_template = load_prompt("/home/kunjanmanoj/Desktop/Doubts_SkolarX/ai_teacher_student/prompts/teacher_prompt.txt")
    prompt = format_prompt(prompt_template, {"concept": concept})

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=300)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def answer_doubt(lesson:str,question: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> str:
    prompt_template = load_prompt("/home/kunjanmanoj/Desktop/Doubts_SkolarX/ai_teacher_student/prompts/teacher_answer_prompt.txt")
    prompt = format_prompt(prompt_template, {"question": question,"concept_knowledge":lesson})

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
