from __future__ import annotations

from pathlib import Path
import argparse
import builtins
import gc
import re

import numpy as np
import torch
from tqdm import tqdm


ROOT = Path.cwd()
DATASET_ROOT = ROOT / "eukaryote_model_datasets"
FEATURE_ROOT = DATASET_ROOT / "features"

ESM_MODEL = Path(r"D:\ESM\esm2_t33_650M_UR50D.pt")
PROTT5_DIR = Path(r"C:\Users\tly\.cache\huggingface\hub\models--Rostlab--prot_t5_xl_uniref50\snapshots\973be27c52ee6474de9c945952a8008aeb2a1a73")
PROTBERT_DIR = Path(r"C:\Users\tly\.cache\huggingface\hub\models--Rostlab--prot_bert\snapshots\7a894481acdc12202f0a415dd567f6cfdb698908")

VALID_AA = set("ACDEFGHIKLMNPQRSTVWYX")


def safe_id(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def clean_seq(seq: str) -> str:
    seq = str(seq).upper().replace("*", "")
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWYX]", "X", seq)


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records = []
    header = ""
    seq_parts = []

    def flush():
        if header:
            rid = header[1:].split()[0]
            records.append((rid, clean_seq("".join(seq_parts))))

    with path.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                flush()
                header = line
                seq_parts = []
            else:
                seq_parts.append(line.strip())
    flush()
    return records


def l2_norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-8)


def sliding_chunks(seq: str, window: int = 1022, stride: int = 768) -> list[str]:
    chunks = []
    for start in range(0, len(seq), stride):
        chunk = seq[start : start + window]
        if len(chunk) > 30:
            chunks.append(chunk)
    return chunks


def load_esm(device: torch.device):
    import esm
    import argparse as _argparse

    torch.serialization.add_safe_globals([_argparse.Namespace])
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(ESM_MODEL))
    return model.to(device).eval(), alphabet.get_batch_converter()


def embed_esm(seq: str, model, batch_converter, device: torch.device, max_len: int) -> np.ndarray:
    seq = clean_seq(seq)[:max_len]
    chunks = sliding_chunks(seq)
    global_list = []
    for chunk in chunks:
        _, _, tokens = batch_converter([("0", chunk)])
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[33])
        rep = out["representations"][33][0, 1:-1]
        g = torch.cat([rep.mean(0), rep.max(0).values]).detach().cpu().numpy()
        global_list.append(g)
        del out, rep, tokens
    if not global_list:
        return np.zeros(2560, dtype=np.float32)
    return l2_norm(np.mean(global_list, axis=0)).astype(np.float32)


def embed_esm_batch(
    seqs: list[str],
    model,
    batch_converter,
    device: torch.device,
    max_len: int,
    chunk_batch_size: int,
) -> list[np.ndarray]:
    chunk_items: list[tuple[int, str]] = []
    for seq_i, seq in builtins.enumerate(seqs):
        chunks = sliding_chunks(clean_seq(seq)[:max_len])
        for chunk in chunks:
            chunk_items.append((seq_i, chunk))

    global_lists: list[list[np.ndarray]] = [[] for _ in seqs]
    start = 0
    current_batch_size = max(1, chunk_batch_size)
    while start < len(chunk_items):
        batch_items = chunk_items[start : start + current_batch_size]
        labels_and_chunks = [(str(i), chunk) for i, (_, chunk) in builtins.enumerate(batch_items)]
        _, _, tokens = batch_converter(labels_and_chunks)
        tokens = tokens.to(device)
        try:
            with torch.no_grad():
                out = model(tokens, repr_layers=[33])
        except RuntimeError as e:
            del tokens
            if "out of memory" in str(e).lower() and torch.cuda.is_available() and current_batch_size > 1:
                torch.cuda.empty_cache()
                current_batch_size = max(1, current_batch_size // 2)
                continue
            raise
        reps = out["representations"][33]
        for j, (seq_i, chunk) in builtins.enumerate(batch_items):
            rep = reps[j, 1 : 1 + len(chunk)]
            g = torch.cat([rep.mean(0), rep.max(0).values]).detach().cpu().numpy()
            global_lists[seq_i].append(g)
        del out, reps, tokens
        start += len(batch_items)

    results = []
    for values in global_lists:
        if values:
            results.append(l2_norm(np.mean(values, axis=0)).astype(np.float32))
        else:
            results.append(np.zeros(2560, dtype=np.float32))
    return results


def load_prott5(device: torch.device):
    from transformers import T5Tokenizer, T5EncoderModel

    tokenizer = T5Tokenizer.from_pretrained(str(PROTT5_DIR), local_files_only=True)
    model = T5EncoderModel.from_pretrained(str(PROTT5_DIR), local_files_only=True)
    return tokenizer, model.to(device).eval()


def embed_t5(seq: str, tokenizer, model, device: torch.device, max_len: int) -> np.ndarray:
    seq = clean_seq(seq)[:max_len]
    seq_spaced = " ".join(seq)
    tokens = tokenizer(seq_spaced, return_tensors="pt", truncation=True, max_length=max_len + 2).to(device)
    with torch.no_grad():
        out = model(**tokens).last_hidden_state.squeeze(0)
    mask = tokens.attention_mask.squeeze(0).bool()
    out = out[mask]
    emb = torch.cat([out.mean(0), out.max(0).values]).detach().cpu().numpy()
    return l2_norm(emb).astype(np.float32)


def embed_t5_batch(
    seqs: list[str],
    tokenizer,
    model,
    device: torch.device,
    max_len: int,
    batch_size: int,
) -> list[np.ndarray]:
    results: list[np.ndarray] = []
    spaced = [" ".join(clean_seq(seq)[:max_len]) for seq in seqs]
    for start in range(0, len(spaced), batch_size):
        batch = spaced[start : start + batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len + 2).to(device)
        with torch.no_grad():
            hidden = model(**tokens).last_hidden_state
        mask = tokens.attention_mask.bool()
        masked = hidden.masked_fill(~mask.unsqueeze(-1), 0.0)
        lengths = mask.sum(dim=1).clamp(min=1).unsqueeze(1)
        mean = masked.sum(dim=1) / lengths
        max_hidden = hidden.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
        emb = torch.cat([mean, max_hidden], dim=1).detach().cpu().numpy()
        results.extend([l2_norm(x).astype(np.float32) for x in emb])
        del tokens, hidden, mask, masked, lengths, mean, max_hidden, emb
    return results


def load_protbert(device: torch.device):
    from transformers import BertTokenizer, BertModel

    tokenizer = BertTokenizer.from_pretrained(str(PROTBERT_DIR), do_lower_case=False, local_files_only=True)
    model = BertModel.from_pretrained(str(PROTBERT_DIR), local_files_only=True)
    return tokenizer, model.to(device).eval()


def embed_protbert(seq: str, tokenizer, model, device: torch.device, max_len: int) -> np.ndarray:
    seq = clean_seq(seq)[:max_len]
    tokens = tokenizer(" ".join(seq), return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model(**tokens).last_hidden_state.squeeze(0)
    out = out[1:-1]
    emb = torch.cat([out.mean(0), out.max(0).values]).detach().cpu().numpy()
    return l2_norm(emb).astype(np.float32)


def embed_protbert_batch(
    seqs: list[str],
    tokenizer,
    model,
    device: torch.device,
    max_len: int,
    batch_size: int,
) -> list[np.ndarray]:
    results: list[np.ndarray] = []
    spaced = [" ".join(clean_seq(seq)[:max_len]) for seq in seqs]
    for start in range(0, len(spaced), batch_size):
        batch = spaced[start : start + batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            hidden = model(**tokens).last_hidden_state
        mask = tokens.attention_mask.bool()
        # Match the previous single-sequence behavior: exclude CLS and SEP.
        valid = mask.clone()
        valid[:, 0] = False
        last_idx = mask.sum(dim=1).clamp(min=1) - 1
        valid[torch.arange(valid.shape[0], device=device), last_idx] = False
        masked = hidden.masked_fill(~valid.unsqueeze(-1), 0.0)
        lengths = valid.sum(dim=1).clamp(min=1).unsqueeze(1)
        mean = masked.sum(dim=1) / lengths
        max_hidden = hidden.masked_fill(~valid.unsqueeze(-1), -1e9).max(dim=1).values
        emb = torch.cat([mean, max_hidden], dim=1).detach().cpu().numpy()
        results.extend([l2_norm(x).astype(np.float32) for x in emb])
        del tokens, hidden, mask, valid, masked, lengths, mean, max_hidden, emb
    return results


def process(dataset: str, model_name: str, max_len: int, splits: list[str], batch_size: int) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)
    split_dir = DATASET_ROOT / dataset / "random_species_label_split_seed42"
    out_dir = FEATURE_ROOT / dataset / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if model_name == "esm2":
        model, helper = load_esm(device)
        embed_fn = lambda s: embed_esm(s, model, helper, device, max_len)
        batch_embed_fn = lambda ss: embed_esm_batch(ss, model, helper, device, max_len, batch_size)
    elif model_name == "prott5":
        helper, model = load_prott5(device)
        embed_fn = lambda s: embed_t5(s, helper, model, device, max_len)
        batch_embed_fn = lambda ss: embed_t5_batch(ss, helper, model, device, max_len, batch_size)
    elif model_name == "protbert":
        helper, model = load_protbert(device)
        embed_fn = lambda s: embed_protbert(s, helper, model, device, max_len)
        batch_embed_fn = lambda ss: embed_protbert_batch(ss, helper, model, device, max_len, batch_size)
    else:
        raise ValueError(model_name)

    for split in splits:
        records = read_fasta(split_dir / f"{split}.protein.fasta")
        per_dir = out_dir / "per_seq" / split
        per_dir.mkdir(parents=True, exist_ok=True)
        ids = np.asarray([rid for rid, _ in records], dtype=object)
        np.save(out_dir / f"{split}_ids.npy", ids, allow_pickle=True)
        pending = [(rid, seq) for rid, seq in records if not (per_dir / f"{safe_id(rid)}.npy").exists()]
        pending.sort(key=lambda x: len(x[1]))
        progress = tqdm(total=len(pending), desc=f"{dataset}-{model_name}-{split}")
        for start in range(0, len(pending), batch_size):
            batch_records = pending[start : start + batch_size]
            try:
                if batch_size <= 1:
                    embs = [embed_fn(seq) for _, seq in batch_records]
                else:
                    embs = batch_embed_fn([seq for _, seq in batch_records])
                for (rid, _), emb in zip(batch_records, embs):
                    np.save(per_dir / f"{safe_id(rid)}.npy", emb)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    for rid, seq in batch_records:
                        emb = embed_fn(seq)
                        np.save(per_dir / f"{safe_id(rid)}.npy", emb)
                else:
                    raise
            progress.update(len(batch_records))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        progress.close()
        gc.collect()
        embs = [np.load(per_dir / f"{safe_id(rid)}.npy") for rid, _ in records]
        arr = np.asarray(embs, dtype=np.float32)
        np.save(out_dir / f"{split}_emb.npy", arr)
        print(dataset, model_name, split, arr.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True, choices=["esm2", "prott5", "protbert"])
    parser.add_argument("--max-len", type=int, default=3000)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    process(args.dataset, args.model, args.max_len, args.splits, args.batch_size)


if __name__ == "__main__":
    main()
