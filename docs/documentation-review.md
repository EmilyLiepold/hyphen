# Independent documentation review

Before advancing past new or revised documentation or examples, the writer must
check whether a fresh agent can understand and apply them without private context.
This procedure also applies to repository instructions. It tests usability;
checking documentation against the implementation and obtaining human acceptance
are separate requirements in [AGENTS.md](../AGENTS.md).

## Procedure

1. **Fix the review target and criteria.** Identify the exact draft, intended
   audience, and essential facts and actions a reader must understand. Prepare a
   short set of open-ended application questions and expected answers grounded in
   the code or policy before reading the reviewer's responses. Include a normal
   use case, a plausible misuse or failure, and a limitation. Keep the criteria
   stable; do not lower them to obtain a pass.
2. **Start an uninvolved reader.** Use a fresh subagent without the author's chat,
   explanations, or previous review attempts. Explicitly select a different,
   smaller model and lower reasoning effort: prefer Luna at low effort when the
   writer uses Sol. In the current collaboration tools, use `fork_turns="none"`,
   `model="gpt-6-luna"`, and `reasoning_effort="low"`. Record the actual settings.
   If the required model or fresh context is unavailable, report the checkpoint
   as incomplete; do not silently substitute the writer or a coached reviewer.
3. **Request an independent explanation first.** Supply only the proposed
   documentation, examples, and explicitly linked prerequisites a normal reader
   would have. Ask the reader to explain the purpose, workflow, inputs/outputs,
   assumptions, limits, and unclear points in its own words. For policy documents,
   explain the actors, decisions, required evidence, and stopping conditions.
   Do not supply the answer key, leading hints, source-code browsing, or an
   explanation that repairs gaps in the draft. Read-only review must not launch
   simulations, change files, or spawn further reviewers.
4. **Quiz the reader in a separate exchange.** After receiving that explanation,
   ask the prepared questions without giving answers or corrective coaching.
   Require concrete steps and reasoning, with documentation references. Ask the
   reader to identify missing information rather than invent it. A fluent summary
   or a generic statement that the documentation is clear is insufficient.
5. **Evaluate against the fixed criteria.** A pass requires a faithful explanation
   and correct, supported answers to every essential question, including failure
   handling and limitations. Unsupported assumptions, material omissions, wrong
   steps or units, and unsafe conclusions fail the review; do not average them
   away with a numerical score. Nonessential wording suggestions need not fail.
6. **Repair the document and retest.** Investigate mistakes against the document
   and underlying code or policy; the reader and writer can both be wrong. Put
   necessary clarifications in the document, then start a new uninvolved reader
   for the revised draft and repeat both exchanges. Do not coach the same reader
   to a passing answer, select a favorable retry on an unchanged draft, or use
   stronger reasoning to compensate for unclear prose. If failures persist, stop
   dependent work and explain the unresolved issue to the human.
7. **Expose the evidence and return to the human.** Keep the draft identity,
   criteria, model/effort, initial explanation, questions, answers, evaluation,
   and revisions in the PR, chat, or another visible review record. Preserve
   failed attempts. A pass permits presenting the documentation as having passed
   this check; it does not authorize commits, production runs, or acceptance.
   Complete the human understanding and approval checkpoints before advancing.

## Example review questions

For this procedure itself, a writer could ask:

- What must change after a reader needs extra explanation before it can answer?
- What should happen if one essential answer is wrong despite a strong summary?
- Which actions are authorized after the reader passes, and which remain gated?

These illustrate application questions, not a reusable answer key. Choose
questions that exercise the actual documentation under review. Reading examples
does not authorize executing them; execution stays within the human-approved
resource budget.

Model settings and fresh-context mechanisms depend on the available tools. See
the [official subagent documentation](https://learn.chatgpt.com/docs/agent-configuration/subagents)
for Codex model and reasoning selection. All repository policy is stated here
and in `AGENTS.md`; no hidden agent configuration is required.
