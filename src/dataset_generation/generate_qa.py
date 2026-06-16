import re, textwrap
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = textwrap.dedent("""
Você é um especialista em criação de datasets QA e Retorne um JSON com Pergunta e Resposta

Exemplo:

Trecho:
ANAC: Agência Nacional de Aviação Civil.

PERGUNTA: O que significa ANAC?
RESPOSTA: ANAC significa Agência Nacional de Aviação Civil.

Agora faça o mesmo para o trecho abaixo.

Trecho:
{chunk}

PERGUNTA:
""").strip()

def load_generator(model_id: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    return model, tok

def generate_pair(model, tokenizer, chunk: str, device: str) -> dict | None:
    prompt = PROMPT.format(chunk=chunk[:600])
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=120, do_sample=False,
            temperature=0.7, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    return _parse(text)

def _parse(text: str) -> dict | None:
    m = re.search(
        r"(?:PERGUNTA|P)\s*[:\-]\s*(.+?)\s*(?:RESPOSTA|R)\s*[:\-]\s*(.+)",
        text, re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return None
    instr, out = m.group(1).strip(), m.group(2).strip()
    if len(instr) < 10 or len(out) < 20:
        return None
    return {"Instruction": instr, "Output": out}