from pathlib import Path
import json

def load_prompt(path: str) -> str:
    with open(path, "r") as f:
        return f.read()

def format_prompt(template: str, values: dict) -> str:
    for key, val in values.items():
        template = template.replace(f"{{{{{key}}}}}", str(val))
    return template

def log_doubt(concept: str, question: str, answer: str, path="/home/kunjanmanoj/Desktop/Doubts_SkolarX/ai_teacher_student/data/doubt_log.jsonl"):
    log = {
        "concept": concept,
        "question": question,
        "answer": answer
    }
    with open(path, "a") as f:
        f.write(json.dumps(log) + "\n")
