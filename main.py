from config.model_config import load_tinyllama_model
from models.teacher import teach_concept, answer_doubt
from models.student import generate_doubt
from utils.io_utils import log_doubt
import json

# Load model
tokenizer, model = load_tinyllama_model()

# Load concepts
with open("data/concept_data.json") as f:
    concepts = json.load(f)

for concept in concepts:
    print(f"\n🔹 Teaching: {concept}")
    lesson = teach_concept(concept, model, tokenizer)
    print(f"\n🧑‍🏫 LESSON:\n{lesson}")

    past_qs = []
    for i in range(1, 4):
        q = generate_doubt(lesson, past_qs, i, model, tokenizer)
        print(f"\n🤔 Doubt {i}:\n{q}")
        past_qs.append(q)

        ans = answer_doubt(lesson,q, model, tokenizer)
        print(f"🧑‍🏫 Answer {i}:\n{ans}")

        log_doubt(concept, q, ans)
