import re

def is_valid(pair: dict) -> bool:
    instr = pair.get("Instruction", "")
    out   = pair.get("Output", "")
    if len(instr.split()) < 5 or len(out.split()) < 10:
        return False
    if instr.lower().strip("?") in out.lower():
        return False
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", instr + out):
        return False
    return True