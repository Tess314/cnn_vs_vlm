# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer, AutoProcessor, TrainingArguments, LlavaForConditionalGeneration, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig

"""### Load the model (quantized)"""

model_id = "llava-hf/llava-1.5-7b-hf"

"""
quantization_config = BitsAndBytesConfig(
    #load_in_2bit=True,
    #bnb_2bit_compute_dtype=torch.float16,
    #load_in_4bit=True,
    #bnb_4bit_compute_dtype=torch.float16,
    #load_in_6bit=True,
    #bnb_6bit_compute_dtype=torch.float16,
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
)
"""

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    #quantization_config=quantization_config,
    #torch_dtype=torch.float16, #causes problems
    torch_dtype=torch.bfloat16,
    #device_map="auto", #causes problems
)

"""### Create a Chat template set `tokenizer` and `processor`"""

LLAVA_CHAT_TEMPLATE = """You are an expert in skin lesion diagnosis.

{% for message in messages %}
{% if message['role'] == 'user' %}
USER: {% for item in message['content'] %}
{% if item['type'] == 'text' %}{{ item['text'] }}
{% elif item['type'] == 'image' %}<image>
{% endif %}{% endfor %}
{% else %}
ASSISTANT: {% for item in message['content'] %}
{% if item['type'] == 'text' %}{{ item['text'] }}
{% endif %}{% endfor %}{{ eos_token }}
{% endif %}
{% endfor %}
"""

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.chat_template = LLAVA_CHAT_TEMPLATE
processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer = tokenizer

"""### Create a `DataCollator`"""

class LLavaDataCollator:
    def __init__(self, processor, image_base_path="/users/tw4001/Documents/train"):
        """
        Args:
            processor: The multimodal processor (from AutoProcessor.from_pretrained()).
            image_base_path: Folder where your training images (ISIC_*.jpg) are stored.
        """
        self.processor = processor
        self.image_base_path = image_base_path

    def __call__(self, examples):
        texts = []
        images = []

        for example in examples:
            messages = example["messages"]

            # Convert chat-style messages to a single text string
            text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

            # Extract image filename from the user message
            img_path = None
            for c in messages[0]["content"]:
                if c["type"] == "image" and c["image_url"] is not None:
                    img_path = os.path.join(self.image_base_path, c["image_url"])
                    break  # only one image per example

            if img_path is not None and os.path.exists(img_path):
                images.append(img_path)
            else:
                # If image missing or not found, append an empty list
                images.append([])

        # Use processor to build the multimodal batch
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True
        )

        # Create labels for supervised fine-tuning
        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch

data_collator = LLavaDataCollator(processor, image_base_path="/Documents/train")

"""### Load the Dataset"""

from datasets import load_dataset

data_files = {
    "train": "data_train_llava.jsonl",
    "test": "data_val_llava.jsonl"
}

dataset = load_dataset("json", data_files=data_files)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

def fix_prompts(example):
    messages = example["messages"]
    user_content = messages[0]["content"]

    for c in user_content:
        if c["type"] == "text":
            text = c["text"]
            # Remove <LABEL> and any leftover <image> in text
            text = text.replace("<LABEL>", "").replace("<image>", "")
            text = text.replace("and image", "").strip()
            c["text"] = text.strip()
    return example

train_dataset = train_dataset.map(fix_prompts)
eval_dataset = eval_dataset.map(fix_prompts)

import os
def add_image_paths(example, image_base_path="/Documents/train"):
    images = []
    for message in example["messages"]:
        for content in message["content"]:
            if content["type"] == "image" and content["image_url"]:
                img_path = os.path.join(image_base_path, content["image_url"])
                images.append(img_path)
    example["images"] = images  # SFTTrainer expects this
    return example

train_dataset = train_dataset.map(add_image_paths)
eval_dataset = eval_dataset.map(add_image_paths)

train_dataset[0]

"""### Set the Training Arguments"""

training_args = TrainingArguments(
    output_dir = "llava-finetuned",
    #overwrite_output_dir=True,
    #report_to="tensorboard",
    learning_rate=0.0002, 
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4, 
    logging_steps=5,
    num_train_epochs=1, 
    push_to_hub=False,
    push_to_hub_token=None,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    fp16=False, 
    bf16=True
)

"""### Set the `LoRA` config"""

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules="all-linear"
)

"""### Create the `SFTTrainer`object


"""

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    data_collator=data_collator,
)

"""### Start the training"""

trainer.train()

### Testing and Evaluation

import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import re

# Normalization helper
def normalize_label(text):
    text = text.strip().lower()
    text = text.replace(".", "").replace("_", " ").replace(",", "")
    return text

# Helper to extract the label properly
def extract_pred_label(text, label_list):

    #Extract the most likely class name from the model's text output.
    #Looks for any known label substring inside the generated text.

    if not text:
        return "unmatched"

    text = text.lower().strip()

    # direct substring match
    for label in label_list:
        if label.lower() in text:
            return label

    # fallback: clean punctuation, fuzzy match
    text_clean = re.sub(r"[^a-z\s]", "", text)
    for label in label_list:
        label_clean = re.sub(r"[^a-z\s]", "", label.lower())
        if label_clean in text_clean:
            return label

    return "unmatched"

# Main evaluation function
def evaluate_llava(model, processor, test_dataset, image_folder, max_new_tokens=30, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    preds, labels = [], []

    # Define your known label list
    label_list = [
        "basal cell carcinoma",
        "benign keratosis",
        "melanoma",
        "nevus",
        "solar or actinic keratosis",
        "squamous cell carcinoma",
        "vascular lesion",
        "scar",
        "unknown"
    ]

    for example in tqdm(test_dataset, desc="Evaluating"):
        messages = example["messages"]

        # Build chat prompt using the same template as training
        text_prompt = processor.tokenizer.apply_chat_template(
            messages[:-1],  # only user messages
            tokenize=False,
            add_generation_prompt=True
        )

        # Find image file path
        image_filename = None
        for c in messages[0]["content"]:
            if c["type"] == "image" and c["image_url"]:
                image_filename = c["image_url"]
                break

        if not image_filename:
            continue

        image_path = os.path.join(image_folder, image_filename)

        # Ground-truth label
        gt_label = messages[1]["content"][0]["text"]

        # Prepare inputs
        inputs = processor(
            text=text_prompt,
            images=image_path,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Clean and map both prediction and label
        pred_label = extract_pred_label(normalize_label(generated_text), label_list)
        true_label = extract_pred_label(normalize_label(gt_label), label_list)

        preds.append(pred_label)
        labels.append(true_label)

    # Evaluation metrics
    accuracy = accuracy_score(labels, preds)
    print(f"\nTest accuracy: {accuracy*100:.2f}%")

    print("\nClassification report:")
    print(classification_report(labels, preds, digits=4))

    print("\nConfusion matrix:")
    print(confusion_matrix(labels, preds))

    return accuracy, preds, labels


# Run evaluation
image_folder = "/Documents/test"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accuracy, preds, labels = evaluate_llava(model, processor, eval_dataset, image_folder, device=device)