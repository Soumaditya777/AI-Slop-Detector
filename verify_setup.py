from peft import PeftConfig, AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer
from datasets import load_dataset
import torch

MODEL_ID = "gouwsxander/slop-detector-bert"
DATASET_ID = "gouwsxander/wikipedia-human-ai"

print("\n1. CHECKING MODEL CONFIGURATION")
adapter_config = PeftConfig.from_pretrained(MODEL_ID)
print("Adapter base model:", adapter_config.base_model_name_or_path)
print("PEFT task:", adapter_config.task_type)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoPeftModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

print("Model class:", type(model).__name__)
print("Number of labels:", model.config.num_labels)
print("Label mapping:", model.config.id2label)
print("Loaded adapter:", list(model.peft_config.keys()))

print("\n2. CHECKING DATASET")
dataset = load_dataset(DATASET_ID)
print("Available splits:", list(dataset.keys()))

for split_name, split in dataset.items():
    print(f"\nSplit: {split_name}")
    print("Rows:", len(split))
    print("Columns:", split.column_names)

    if len(split) > 0:
        print("First row keys:", list(split[0].keys()))

# This dataset uses paired columns, not a separate label column.
data = dataset["train"]
example = data[0]

print("\n3. TESTING A PAIRED EXAMPLE")
print("Page title:", example["page_title"])
print("Human text preview:", example["human_text"][:150])
print("AI text preview:", example["ai_text"][:150])

device = (
    "mps"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
model.to(device)
print("Inference device:", device)


@torch.inference_mode()
def score(text):
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    logits = model(**inputs).logits
    probs = torch.softmax(logits.float(), dim=-1)[0]
    return probs.tolist()


human_probs = score(example["human_text"])
ai_probs = score(example["ai_text"])

print("\n4. AI-CLASS SCORES")
print("Human paragraph — LABEL_1 score:", round(human_probs[1] * 100, 2), "%")
print("AI rewrite      — LABEL_1 score:", round(ai_probs[1] * 100, 2), "%")

print("\nVerification complete.")