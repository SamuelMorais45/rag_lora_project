def chunk_text(text: str, max_length: int = 500, overlap: int = 50) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start : start + max_length].strip()
        if len(chunk) >= 50:
            chunks.append(chunk)
        start += max_length - overlap
    return chunks