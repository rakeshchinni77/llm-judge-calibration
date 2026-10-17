# Annotation Rubric — Technical Question Answering Evaluation

**Version:** 2.0
**Author:** [Your name / handle]
**Task:** Evaluate whether an AI-generated answer to a technical question is a
"Pass" or a "Fail" according to the criteria defined below.
**Updated:** Phase 1 — task changed from GitHub issue summarisation to Technical Q&A.

---

## Task Definition

A language model is given a **technical question** drawn from software engineering,
Python programming, algorithms, machine learning, statistics, or databases. The
model is asked to provide a concise, accurate answer suitable for an engineer
looking for a quick explanation or refresher.

Generation conditions are deliberately varied (temperature, prompt completeness)
so the dataset contains a realistic mix of passing and failing answers, with a
target of **at least 30 % failures**.

---

## What Counts as a PASS (label = 1)

An answer earns a **Pass** if it satisfies **all** of the following:

| # | Criterion | Description |
|---|-----------|-------------|
| P1 | **Technical correctness** | Every factual claim is verifiably correct per authoritative references (Python docs, CLRS, standard ML texts). No incorrect formulas, wrong complexities, or wrong syntax. |
| P2 | **Addresses the question** | The answer directly responds to what was asked. It does not answer a different or simpler question. |
| P3 | **Key concepts present** | Core concepts required to correctly answer the question are included. Minor optional elaborations may be absent without causing a Fail. |
| P4 | **No harmful fabrication** | The answer does not invent APIs, algorithms, or terminology that do not exist. |
| P5 | **Readable and complete** | The text is coherent English with no truncation or obvious incoherence. |

> An answer may be brief or verbose and still Pass, as long as the content is
> correct and the question is addressed.

---

## What Counts as a FAIL (label = 0)

An answer earns a **Fail** if it violates **any** of the following:

| # | Criterion | Description |
|---|-----------|-------------|
| F1 | **Factual error** | States something objectively false (wrong complexity, wrong syntax, incorrect definition, wrong formula). |
| F2 | **Does not address the question** | Answers a related but different question, or deflects entirely. |
| F3 | **Critical concept missing** | Omits a concept without which the answer is misleading (e.g., explaining gradient descent without mentioning the learning rate). |
| F4 | **Fabrication** | Invents a function, library, algorithm name, or terminology that does not exist. |
| F5 | **Incoherent or truncated** | Text is grammatically broken, self-contradictory, or ends mid-sentence. |

---

## Edge Cases and Tie-Breaking Rules

| Situation | Decision |
|-----------|----------|
| Answer is technically correct but unusually brief (1 sentence) | **Pass** if no critical concept is missing |
| Answer is verbose with mostly correct content but one minor error | **Fail** (F1) — correctness is non-negotiable |
| Answer hedges ("I believe…", "typically…") but content is correct | **Pass** — hedging alone does not cause failure |
| Answer gives correct concept with a slightly wrong code example | **Fail** (F1) — code that will not run or gives wrong results is an error |
| Answer is correct for a related but different version (e.g., Python 2 syntax) | **Fail** (F2) — the question context implies Python 3 unless stated |
| Genuinely ambiguous; you are unsure after 60 seconds | Label **Fail** (conservative bias) and write a critique noting the ambiguity |
| Answer introduces caveats that are correct but not asked for | **Pass** — extra correct context is fine |
| Answer is partially correct but the wrong part is load-bearing | **Fail** (F1 or F3) |

---

## Labeling Protocol

1. Read the **question** (the `question` column in the annotation template — your
   context for judging correctness).
2. Read the **generated answer** (the `output` column — what you are evaluating).
3. Check each Pass criterion (P1–P5). If all are met → label `1`.
4. Check each Fail criterion (F1–F5). If any is triggered → label `0`.
5. Write a **1–3 sentence critique** explaining your decision:
   - **Pass:** State which criterion it best demonstrates and why it's correct.
   - **Fail:** Cite the specific criterion violated with a short quote or example.
6. Set `relabeled = 0` initially. If you later change a label after reading the
   LLM judge's critique, update the label and set `relabeled = 1`.
7. **Conservative tie-breaking:** When in genuine doubt, prefer Fail.

---

## Reference Examples

### Example PASS

> **Question:** What is the time complexity of lookup in a Python dict?
>
> **Answer:** "Dictionary lookups in Python have O(1) average-case time complexity
> because Python dicts are implemented as hash tables. In the worst case (many
> hash collisions), lookups can degrade to O(n), but this is extremely rare in
> practice with Python's hash randomisation."
>
> **Label:** 1 — Correct average and worst-case complexities stated, implementation
> detail (hash table) given, practical note included. Satisfies P1–P5.

### Example FAIL — Factual Error (F1)

> **Question:** What is the time complexity of lookup in a Python dict?
>
> **Answer:** "Python dictionaries use a sorted array internally, so lookup
> is O(log n) using binary search."
>
> **Label:** 0 — Violates F1. Python dicts are hash tables, not sorted arrays;
> lookup is O(1), not O(log n). Fabricates the implementation detail.

### Example FAIL — Missing Critical Concept (F3)

> **Question:** Explain the bias-variance tradeoff in machine learning.
>
> **Answer:** "Bias is the error from overly simple models. Variance is the
> error from overly complex models. You want to minimise both."
>
> **Label:** 0 — Violates F3. The answer does not explain that reducing one
> typically increases the other (the tradeoff), which is the core of the concept.

### Example FAIL — Fabrication (F4)

> **Question:** How do you perform a group-by aggregation in Pandas?
>
> **Answer:** "Use `df.aggregate_by('column').sum()` to group and sum."
>
> **Label:** 0 — Violates F4. `aggregate_by()` does not exist in Pandas;
> the correct method is `df.groupby('column').sum()`.

---

## Trust Thresholds (pre-registered before the frozen test run)

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| TPR lower CI bound | ≥ 0.70 | Catches ≥ 70 % of actual failures |
| TNR lower CI bound | ≥ 0.80 | Correctly passes ≥ 80 % of good answers |
| Kappa lower CI bound | ≥ 0.60 | Substantial inter-rater agreement |

These thresholds are **frozen** and must not be adjusted after the test split is run.
