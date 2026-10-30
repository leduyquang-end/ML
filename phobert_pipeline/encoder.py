"""README mục 6: PhoBERT đóng băng và masked mean pooling cho từng chunk."""

from .config import CHECKPOINT, MODEL_CACHE, configure_runtime


def load_encoder(seed: int):
    """Nạp encoder/tokenizer từ cache ổ E và tắt cập nhật trọng số/dropout."""
    configure_runtime()
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, cache_dir=MODEL_CACHE, use_fast=False)
    encoder = AutoModel.from_pretrained(CHECKPOINT, cache_dir=MODEL_CACHE).to(device)
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder, tokenizer, device


def encode_batch(chunks, encoder, tokenizer, limit, device):
    """Trả ma trận [số chunk, hidden_size], pooling chỉ trên token nội dung."""
    import torch

    # Thêm <s>, </s> riêng cho từng chunk SAU khi chia token nội dung.
    prepared = [tokenizer.prepare_for_model(
        tokens, add_special_tokens=True, truncation=False,
        return_special_tokens_mask=True, return_token_type_ids=False,
    ) for tokens, _ in chunks]
    if any(len(item["input_ids"]) > limit for item in prepared):
        raise ValueError("Chunk vượt giới hạn vị trí của PhoBERT.")
    batch = tokenizer.pad(prepared, padding=True, return_tensors="pt").to(device)
    special = batch.pop("special_tokens_mask").bool()
    # attention_mask bỏ padding; special_tokens_mask bỏ <s>, </s>.
    # Thêm chiều cuối để nhân mask [B,T,1] với hidden state [B,T,H].
    mask = (batch["attention_mask"].bool() & ~special).unsqueeze(-1)
    with torch.no_grad():
        hidden = encoder(**batch).last_hidden_state
        vectors = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
    return vectors.cpu().numpy()
