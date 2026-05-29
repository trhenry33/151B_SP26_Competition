Prompt templates to ensure grader-friendly outputs

Guidelines (must follow):
- If you use chain-of-thought, wrap reasoning in `<think>...</think>` and place the final boxed answer after the last `</think>` tag.
- The final answer must appear exactly as a single `\boxed{...}` on its own line with no trailing text.
- For MCQ, put the single letter inside the box, e.g. `\boxed{F}`.
- For expressions or numeric answers, use parseable LaTeX or plain arithmetic, e.g. `\boxed{325*(1+325)}` or `\boxed{\frac{5}{8}}`.
- Use deterministic decoding for MCQ: `temperature=0`, `do_sample=False`.

Free-form (numeric/expression) template:

Solve the following problem.

<think>
Show your step-by-step reasoning clearly but concisely. Keep calculations short; do not restate the problem.
</think>
\boxed{<final answer in LaTeX or arithmetic>}

Example:
<think>
The nth even number is 2n, so the sum of first n evens = n(n+1). For n=325, sum = 325*(325+1).
</think>
\boxed{325*(1+325)}

MCQ template:

Read the problem and show a short justification inside `<think>...</think>`, then output only the chosen option letter inside the box.

<think>
Eliminate options by checking sign and magnitude; option F matches the computed result.
</think>
\boxed{F}

Two-stage option (safer when reasoning is long):
1) Ask the model for reasoning and stop it (allow longer max tokens).
2) Run a second short prompt that asks: "Based on the reasoning above, output only the final answer inside \boxed{...} (no other text)."


Notes:
- The grader extracts only content after the last `</think>` and prioritizes `\boxed{...}`. Make the box the last thing the model outputs.
- Avoid phrases like "Answer:" or "Final:" outside the box — they confuse strict extraction.
