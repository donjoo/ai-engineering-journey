import re
from collections import Counter

problem = ("A store has 23 apples. It sells 9, then receives 3 boxes with 12 apples each. "
           "Then it sells 15 more. How many apples does it have now?")

def extract_answer(text):
    matches = re.findall(r"Answer:\s*(-?\d+)", text)
    return matches[-1] if matches else None