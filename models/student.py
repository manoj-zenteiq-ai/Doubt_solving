from utils.io_utils import load_prompt, format_prompt
from transformers import PreTrainedModel, PreTrainedTokenizer

def generate_doubt(lesson: str, past_questions: list[str], i: int, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> str:
    prompt_template = load_prompt("/home/kunjanmanoj/Desktop/Doubts_SkolarX/ai_teacher_student/prompts/student_doubt_prompt.txt")
    prompt = format_prompt(prompt_template, {
        "lesson": lesson,
        "past_questions": str(past_questions),
        "i": i
    })

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
