# Few-Shot Example Template
# ---------------------------------------------------------------------------
# This template is populated by src/judge.py using items from data/splits/train.csv.
# Each example is formatted as a user turn followed by an assistant turn.
# ---------------------------------------------------------------------------

## Example {index}: {label_text}

**Summary to evaluate:**
{output}

**Expected response:**
```json
{
  "critique": "{critique}",
  "pass": {pass_value}
}
```
