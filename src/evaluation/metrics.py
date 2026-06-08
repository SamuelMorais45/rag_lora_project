import math, torch
from src.utils.helpers import build_prompt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_response(model, tokenizer, instruction, model_type, max_new=200):
    if model_type == "causal":
        prompt = build_prompt(instruction)
        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new,
                                 do_sample=True, temperature=0.7, top_p=0.9,
                                 pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
    else:
        inputs = tokenizer(instruction, return_tensors="pt",
                           truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new)
        return tokenizer.decode(out[0], skip_special_tokens=True).strip()

def compute_ppl(model, tokenizer, sample, model_type):
    instr, ref = sample["Instruction"], sample["Output"]
    if model_type == "causal":
        prompt = build_prompt(instr)
        full   = prompt + ref
        enc    = tokenizer(full, return_tensors="pt",
                           truncation=True, max_length=512).to(DEVICE)
        plen   = tokenizer(prompt, return_tensors="pt",
                           max_length=512)["input_ids"].shape[1]
        labels = enc["input_ids"].clone()
        labels[0, :plen] = -100
    else:
        enc = tokenizer(instr, return_tensors="pt",
                        truncation=True, max_length=512).to(DEVICE)
        tgt = tokenizer(ref,  return_tensors="pt",
                        truncation=True, max_length=256).to(DEVICE)
        labels = tgt["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        enc = {**enc, "decoder_input_ids": tgt["input_ids"]}
    with torch.no_grad():
        loss = model(**enc, labels=labels).loss
    return math.exp(min(loss.item(), 20))