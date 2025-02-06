"""
***** Finetuning using a Trainer class from the Huggingface Transformers
***** library.
"""
# Imports
import numpy as np
import evaluate
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM

# Global variables
FILENAME = "ja1-train.txt"
VALID_FILE = "ja1-valid.txt"
MODEL = "mistralai/Mistral-7B-v0.1"
LOGIN_TOKEN = "" # Put login token here
LIST = []
VALID_LIST = []

# Logging to HuggingFace
from huggingface_hub import login
login(token=LOGIN_TOKEN)

print("Login successful.")

# Reading text data
f = open(FILENAME)
line = f.readline()
while line:
    LIST += line;
    line = f.readline()
f.close()

print("Read text file " + FILENAME + ".")

f = open(VALID_FILE)
line = f.readline()
while line:
    VALID_LIST += line;
    line = f.readline()
f.close()

print("Read valid file " + VALID_FILE + ".")

# Initializing tokenizer and preprocessing input, etc.
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Bugfix for padding issues: https://discuss.huggingface.co/t/mistral-trouble-when-fine-tuning-dont-set-pad-token-id-eos-token-id/77928/8
tokenizer.add_special_tokens({'pad_token': '<pad>'})

encoded_input = tokenizer(LIST,
                          padding = True,
                          #truncation = True,
                          return_tensors = "pt")

print("Preprocessed training set.")

encoded_valid = tokenizer(VALID_LIST,
                          padding = True,
                          #truncation = True,
                          return_tensors = "pt")

print("Preprocessed validation set.")

# Preparing for training and setting eval function
args = TrainingArguments(output_dir = "test_trainer", eval_strategy = "epoch")
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

print("Checkpoint.")

# Retrieving pretrained model
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map = "auto", torch_dtype="auto")

# Bugfix for padding issues: https://discuss.huggingface.co/t/mistral-trouble-when-fine-tuning-dont-set-pad-token-id-eos-token-id/77928/8
model.resize_token_embeddings(len(tokenizer))

print("Retrieved model.")

# Trainer object
trainer = Trainer(
    model = model,
    args = args,
    train_dataset = encoded_input,
    eval_dataset = encoded_valid,
    compute_metrics = compute_metrics,
)

print("Created Trainer object.")

trainer.train()
