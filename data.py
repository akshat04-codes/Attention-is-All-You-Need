"""
Data Pipeline — Multi30k German→English Translation
=====================================================
Handles downloading, tokenization, vocabulary building, and batching
for the Multi30k dataset.

We use torchtext's built-in Multi30k dataset and spaCy tokenizers.
This is a standard small-scale MT dataset suitable for reproducing
Transformer results on limited compute.
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

# ---------------------------------------------------------------------------
# Lightweight BPE-style word-level tokenizer (no spaCy dependency needed)
# ---------------------------------------------------------------------------


class SimpleTokenizer:
    """
    A simple whitespace + punctuation tokenizer.
    Avoids heavy dependencies like spaCy for portability on Colab.
    """

    def __init__(self):
        import re
        self._pattern = re.compile(r"[\w]+|[^\s\w]", re.UNICODE)

    def tokenize(self, text: str) -> list[str]:
        return self._pattern.findall(text.lower())


class Vocabulary:
    """
    Token-to-index vocabulary built from a corpus.

    Special tokens:
        <pad> = 0, <sos> = 1, <eos> = 2, <unk> = 3
    """

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.token2idx: dict[str, int] = {}
        self.idx2token: dict[int, str] = {}
        self._built = False

    def build(self, sentences: list[list[str]]):
        """Build vocabulary from tokenized sentences."""
        counter = Counter()
        for tokens in sentences:
            counter.update(tokens)

        specials = ["<pad>", "<sos>", "<eos>", "<unk>"]
        self.token2idx = {tok: idx for idx, tok in enumerate(specials)}

        idx = len(specials)
        for token, freq in counter.most_common():
            if freq >= self.min_freq:
                self.token2idx[token] = idx
                idx += 1

        self.idx2token = {v: k for k, v in self.token2idx.items()}
        self._built = True

    def encode(self, tokens: list[str]) -> list[int]:
        """Convert tokens to indices, wrapping with <sos> and <eos>."""
        indices = [self.SOS_IDX]
        indices += [self.token2idx.get(t, self.UNK_IDX) for t in tokens]
        indices.append(self.EOS_IDX)
        return indices

    def decode(self, indices: list[int]) -> list[str]:
        """Convert indices back to tokens, filtering special tokens."""
        special = {self.PAD_IDX, self.SOS_IDX, self.EOS_IDX}
        return [self.idx2token.get(i, "<unk>") for i in indices if i not in special]

    def __len__(self) -> int:
        return len(self.token2idx)


class TranslationDataset(Dataset):
    """
    Generic parallel translation dataset.

    Args:
        src_sentences: List of tokenized source sentences.
        tgt_sentences: List of tokenized target sentences.
        src_vocab: Source vocabulary.
        tgt_vocab: Target vocabulary.
        max_len:   Maximum sequence length (truncate longer).
    """

    def __init__(
        self,
        src_sentences: list[list[str]],
        tgt_sentences: list[list[str]],
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        max_len: int = 128,
    ):
        self.src_data = []
        self.tgt_data = []

        for src_toks, tgt_toks in zip(src_sentences, tgt_sentences):
            src_ids = src_vocab.encode(src_toks[:max_len - 2])
            tgt_ids = tgt_vocab.encode(tgt_toks[:max_len - 2])
            self.src_data.append(torch.tensor(src_ids, dtype=torch.long))
            self.tgt_data.append(torch.tensor(tgt_ids, dtype=torch.long))

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]


def collate_fn(batch):
    """Pad sequences in a batch to the same length."""
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=Vocabulary.PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=Vocabulary.PAD_IDX)
    return src_padded, tgt_padded


# ---------------------------------------------------------------------------
# Multi30k loader (downloads the dataset)
# ---------------------------------------------------------------------------

def _download_multi30k(data_dir: str = "data/multi30k"):
    """Download Multi30k dataset files if not present."""
    os.makedirs(data_dir, exist_ok=True)

    base_url = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"
    files = {
        "train.de": "train.de.gz",
        "train.en": "train.en.gz",
        "val.de": "val.de.gz",
        "val.en": "val.en.gz",
        "test.de": "test_2016_flickr.de.gz",
        "test.en": "test_2016_flickr.en.gz",
    }

    import urllib.request
    import gzip

    for local_name, remote_name in files.items():
        local_path = os.path.join(data_dir, local_name)
        if os.path.exists(local_path):
            continue

        gz_path = local_path + ".gz"
        url = base_url + remote_name
        print(f"Downloading {url}...")
        try:
            urllib.request.urlretrieve(url, gz_path)
            with gzip.open(gz_path, "rt", encoding="utf-8") as f_in:
                with open(local_path, "w", encoding="utf-8") as f_out:
                    f_out.write(f_in.read())
            os.remove(gz_path)
        except Exception as e:
            print(f"  Warning: Could not download {url}: {e}")
            print("  Generating synthetic parallel data for demonstration...")
            _generate_synthetic_data(data_dir)
            return

    print("Multi30k dataset ready.")


def _generate_synthetic_data(data_dir: str):
    """Generate small synthetic parallel corpus for demo when Multi30k is unavailable."""
    import random
    random.seed(42)

    # Simple number-to-word translation task (deterministic, good for testing)
    de_words = ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs",
                "sieben", "acht", "neun", "zehn", "hund", "katze", "haus",
                "baum", "auto", "buch", "tisch", "stuhl", "tür"]
    en_words = ["zero", "one", "two", "three", "four", "five", "six",
                "seven", "eight", "nine", "ten", "dog", "cat", "house",
                "tree", "car", "book", "table", "chair", "door"]

    templates_de = [
        "ich sehe {w1} und {w2}",
        "das ist ein {w1}",
        "{w1} {w2} {w3} sind hier",
        "wo ist der {w1}",
        "der {w1} ist groß",
        "{w1} und {w2} sind klein",
    ]
    templates_en = [
        "i see {w1} and {w2}",
        "that is a {w1}",
        "{w1} {w2} {w3} are here",
        "where is the {w1}",
        "the {w1} is big",
        "{w1} and {w2} are small",
    ]

    for split, n_samples in [("train", 5000), ("val", 500), ("test", 500)]:
        de_lines = []
        en_lines = []
        for _ in range(n_samples):
            idx = random.randint(0, len(templates_de) - 1)
            w_idx = random.sample(range(len(de_words)), 3)
            de_sent = templates_de[idx].format(
                w1=de_words[w_idx[0]], w2=de_words[w_idx[1]], w3=de_words[w_idx[2]]
            )
            en_sent = templates_en[idx].format(
                w1=en_words[w_idx[0]], w2=en_words[w_idx[1]], w3=en_words[w_idx[2]]
            )
            de_lines.append(de_sent)
            en_lines.append(en_sent)

        with open(os.path.join(data_dir, f"{split}.de"), "w", encoding="utf-8") as f:
            f.write("\n".join(de_lines))
        with open(os.path.join(data_dir, f"{split}.en"), "w", encoding="utf-8") as f:
            f.write("\n".join(en_lines))

    print(f"Generated synthetic data in {data_dir}")


def load_multi30k(
    data_dir: str = "data/multi30k",
    batch_size: int = 64,
    max_len: int = 128,
    min_freq: int = 2,
    src_lang: str = "de",
    tgt_lang: str = "en",
) -> tuple[DataLoader, DataLoader, DataLoader, Vocabulary, Vocabulary]:
    """
    Load Multi30k dataset and create DataLoaders.

    Args:
        data_dir: Directory where dataset is saved.
        batch_size: Batch size.
        max_len: Maximum sequence length.
        min_freq: Minimum token frequency for vocabulary.
        src_lang: Source language ("de" or "en").
        tgt_lang: Target language ("en" or "de").

    Returns:
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    """
    _download_multi30k(data_dir)

    tokenizer = SimpleTokenizer()

    # Read and tokenize
    def read_file(path):
        with open(path, "r", encoding="utf-8") as f:
            return [tokenizer.tokenize(line.strip()) for line in f if line.strip()]

    train_src = read_file(os.path.join(data_dir, f"train.{src_lang}"))
    train_tgt = read_file(os.path.join(data_dir, f"train.{tgt_lang}"))
    val_src = read_file(os.path.join(data_dir, f"val.{src_lang}"))
    val_tgt = read_file(os.path.join(data_dir, f"val.{tgt_lang}"))
    test_src = read_file(os.path.join(data_dir, f"test.{src_lang}"))
    test_tgt = read_file(os.path.join(data_dir, f"test.{tgt_lang}"))

    # Build vocabularies from training data
    src_vocab = Vocabulary(min_freq=min_freq)
    src_vocab.build(train_src)
    tgt_vocab = Vocabulary(min_freq=min_freq)
    tgt_vocab.build(train_tgt)

    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(tgt_vocab)}")

    # Create datasets
    train_ds = TranslationDataset(train_src, train_tgt, src_vocab, tgt_vocab, max_len)
    val_ds = TranslationDataset(val_src, val_tgt, src_vocab, tgt_vocab, max_len)
    test_ds = TranslationDataset(test_src, test_tgt, src_vocab, tgt_vocab, max_len)

    # Create dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
