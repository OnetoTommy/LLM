import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# -------------------------
# Load training and test data
# -------------------------
train_df = pd.read_csv("train.txt", header=None, names=["text"])
train_corpus = train_df["text"].dropna().astype(str).tolist()  # clean and ensure str
test_sentence = "he is unfair and unaware and unresponsive"

# -------------------------
# Hyper-parameters
# -------------------------
param_sets = [
    {"min_freq": 2, "vocab_size": 20},
    {"min_freq": 2, "vocab_size": 50},
    {"min_freq": 3, "vocab_size": 50},
]


# -------------------------
# Define training function
# -------------------------
def train_bpe(corpus, min_freq, vocab_size):
    # Initialize empty BPE tokenizer with UNK token
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # Pre-tokenizer ensures words are split by whitespace before BPE merges
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Define trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=["[UNK]", "[PAD]"]
    )

    # Train tokenizer on given corpus
    tokenizer.train_from_iterator(corpus, trainer)
    return tokenizer


# -------------------------
# Train and compare models
# -------------------------
for params in param_sets:
    tokenizer = train_bpe(train_corpus, params["min_freq"], params["vocab_size"])
    output = tokenizer.encode(test_sentence)

    print(f"\n### min_freq={params['min_freq']}, vocab_size={params['vocab_size']}")
    print("Tokens:", output.tokens)
    print("IDs:   ", output.ids)
    print("Number of tokens:", len(output.tokens))
