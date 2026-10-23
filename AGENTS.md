# Agent instructions

This repository implements hyphen, a Python package for fitting and removing
spectral wiggles from JWST/NIRSpec data using detector trace geometry and
reference spectra. It also combines corrected cubes. Changes must earn human
trust through understandable code and scientific evidence.

> A computer can never be held accountable. Therefore a computer must never make
> a management decision.

## Human accountability

- Humans decide scientific goals, scope, acceptance criteria, resource
  commitments, and whether a change is accepted. Agents may investigate, prepare
  draft patches, and run bounded checks within the requested task. An agent must
  never approve its own work or claim authority to accept it.
- Every change to code, tests, configuration, or documentation requires personal
  review by an accountable human before acceptance or production use. Automated
  checks and other agents' reviews provide evidence; they cannot replace that
  review. Record the reviewer, exact revision or diff, validation evidence, and
  explicit approval in the PR, this chat, or another concise review record. Never
  invent approval or treat an implementation request as approval of the result.
- At each review checkpoint, check the human's understanding before advancing
  beyond the current increment. Explain the purpose, changed behavior,
  assumptions, evidence, and limits, with relevant code links or examples. Ask a
  focused, open-ended question about the change and wait for the human's reasoning
  in their own words: for example, how an input affects the result or why a test
  supports the scientific claim. Match the depth to the change; do not quiz them
  on unrelated details or repeat this after routine read-only steps.
- Silence, "looks good," trust in the agent, or generic approval does not establish
  understanding. If the result is being treated as magic or its relevant behavior
  and evidence remain unexplained, pause dependent work. Walk through an example,
  clarify or simplify the change, and check again. Resolve disagreements by
  rechecking the code, evidence, and the agent's own explanation; do not presume
  that the human is mistaken.
- Keep the comprehension exchange in the review record. It provides observable
  evidence of understanding, not certification of competence or permission from
  the agent. Explicit human approval is still required. Never invent a human
  response or claim direct knowledge of their internal mental state.
- Leave patches uncommitted for review. Commit, amend, merge, push, or publish
  only when a human explicitly authorizes that action for the exact reviewed
  change. Subsequent edits require review of the changed content.
- Git authors and committers must be real, accountable people whose attribution
  is confirmed by the authorizing human. Never use an LLM, agent, or bot as an
  author, committer, or `Co-authored-by` identity. Never fabricate or change human
  identity or treat configured Git identity as proof of review. Describe agent
  assistance honestly in review notes; human attribution must reflect an actual
  acceptance of responsibility.

## Work within human review capacity

- Make one coherent, narrowly scoped change at a time. Prefer the smallest clear
  implementation that solves an identified need. Do not bundle unrelated fixes,
  reformatting, speculative features, dependencies, or language migrations.
- Do not generate hundreds of lines ahead of human review. Split larger tasks
  into independently understandable increments, finish the evidence and
  explanation for one, and obtain human review before expanding to the next.
  Do not split files or commits merely to disguise the size of a change.
- When delegating, assign distinct file ownership and scratch/output paths.
  Subagents share these limits; their output counts toward the total review
  burden.

## Git hygiene and change boundaries

- Start every new independent change from a clean Git state. Check the index,
  working tree, and untracked files, e.g.
  `git status --porcelain=v1 --untracked-files=all --ignore-submodules=none`.
  Require no changes and no unfinished merge, rebase, or cherry-pick. Record the
  intended base branch and commit before editing.
- Preserve existing work. Never stash, reset, clean, discard, or commit existing
  work merely to obtain a clean state. If unrelated work is present,
  stop editing that checkout; use a separate clean worktree from the agreed base
  or ask the human to resolve the starting state. Continuing the same identified,
  authorized change is allowed; disclose its existing diff and do not relabel
  unrelated work as a continuation. At handoff, check Git status and review the
  full working-tree diff, including new files.
- Each commit must contain one coherent change. Its directly necessary tests and
  documentation belong with it. Stage only the intended paths or hunks and
  inspect the staged diff before any human-authorized commit. Leave unrelated
  fixes, refactors, formatting, and dependency updates out; handle them separately
  only if requested.
- Each PR must have exactly one independently reviewable purpose. Never combine
  separate changes in one PR, even as separate commits or a squash commit. Keep
  related implementation, tests, and documentation together; use separate branches
  and PRs for independent work. Inspect the entire PR diff against its intended
  target branch to ensure it does not include another change's commits or files.

## Scientific justification and validation

- Before changing behavior, state the scientific or operational need, intended
  result, and proposed acceptance criteria. For each test, explain the failure
  mode or scientific invariant it checks and how its result bears on the change.
  Test counts and passing checks do not by themselves justify a change.
- Trace the actual code, inputs, units, and readers/writers. Preserve WCS
  transforms, wavelength/flux/error units, axis order and pixel indexing,
  reference apertures, wavelength and data-quality masks, model basis and fit
  orders, normalization, parameter bounds, and fit-acceptance criteria unless the
  task explicitly changes them. Explain each intentional change to these
  scientific or interface contracts, including uncertainty propagation and
  preservation of real spectral features.
- For numerical comparisons, use matched, identified inputs and a trusted
  reference. Propose justified tolerances before examining candidate results;
  humans decide whether those tolerances are scientifically acceptable. Retain
  commands, input checksums, resolved configuration, revision and local changes,
  Python/dependency versions, platform and thread/process settings, relevant
  seeds, execution order, and output paths sufficient to reproduce the comparison.
- Never relax thresholds, hide failed fits, filter away non-finite results, or
  change a reference merely to obtain a pass. Report failures and denominators.
  Distinguish installation/import checks, successful execution, analytic tests,
  preserved-output regression, spectral-feature preservation, and uncertainty
  validation; none establishes all the others.
- Match validation to the change: factual and diff review for documentation;
  targeted tests for behavioral changes; relevant scientific comparisons for
  numerical changes. Add regression tests that protect actual behavior or
  invariants, rather than mirror the implementation.
- Keep routine checks bounded and use disposable inputs and outputs. Full
  observing-program runs, large parameter searches, and long benchmarks require
  explicit human authorization of their purpose, scale, and resource budget.
  Small-fixture success does not authorize production use.

## Code and documentation quality

- Be allergic to AI slop. Every added function, abstraction, dependency, comment,
  and document must serve a concrete need. Remove boilerplate, redundant wrappers,
  speculative scaffolding, unsupported claims, and prose that obscures the code.
- Make units, numerical kinds, assumptions, boundary behavior, and failure
  handling explicit where they affect correctness. Do not silently approximate,
  swallow errors, or hide missing validation behind fallback values.
- Document the rationale, interfaces, scientific conventions, reproduction steps,
  and limitations needed to understand and maintain the change. Check prose
  against actual code and commands. Avoid narrating obvious operations or
  claiming work that was not done. Keep documentation concise and reviewable.

## Visible, shared documentation

- Keep repository rules and durable documentation in visible files intended for
  version control: `AGENTS.md`, READMEs, and ordinary documentation directories.
  Link supporting guidance from `AGENTS.md` or a README, directly or through a
  visible documentation index. Maintain one authoritative source for each rule.
- Never create independent repository rules or authoritative documentation in
  `.codex/`, `.claude/`, other hidden directories, ignored files, or private agent
  memory. A necessary tool-specific entry point may only refer to the visible
  source; it must not add policy or maintain a competing copy.
- Document new code for both humans and agents entering with no prior context.
  Include the purpose, prerequisites, setup, interfaces, input/output formats and
  units, assumptions, and limitations needed to use it directly. Required
  operating knowledge must not exist only in the originating chat or agent state.
- Include a minimal, bounded example with small sample inputs or clearly
  identified required data, the exact command or call, expected outputs, and how
  to judge success. Explain paths and dependencies; mark placeholders explicitly.
  Check examples against the implementation and exercise them within the
  authorized resource budget. Label unexecuted examples and their blockers;
  distinguish expected results from observed results.
- Before advancing past a documentation change, follow the
  [independent documentation review](docs/documentation-review.md): a fresh,
  uninvolved reader using a different, smaller model and lower reasoning effort
  must explain the documentation and answer the writer's application questions.
  Revise and repeat until the stated criteria pass. This is a readability check;
  the human understanding checkpoint and acceptance decision remain mandatory.

## Repository-specific checks

- The package contains single/batch IFU and MSA processing paths and frame
  merging. Check the documented support and validation status of the relevant
  workflow; the existence of a code path does not establish that it is supported.
  Do not infer license permissions from a package classifier or release readiness
  from a branch name or version string.
- Use the source map below to locate the behavior before editing. Check actual
  Python/JWST dependency compatibility; package metadata alone is not validation.
  Discover available tests and CI from the current tree rather than assuming a
  test command exists or claiming checks that were not run.

  | Path | Responsibility |
  | --- | --- |
  | `pyproject.toml`, `hyphen/__init__.py` | Dependencies, CLI registration, package version |
  | `hyphen/hyphenate.py`, `hyphen/yaml_helpers.py` | YAML CLI, step dispatch and option layering |
  | `hyphen/*_step.py` | Single/batch IFU/MSA defaults and required options |
  | `hyphen/dewiggler_stack.py` | FITS/WCS, references, orchestration, retries, plots and saved fits |
  | `hyphen/dewiggle_lambda_space.py` | Spectral model, Jacobian, fitting and fit plots |
  | `hyphen/frame_merger.py` | Frame/detector masks, gap handling, caching and weighted merging |
  | `hyphen/tracer_helpers.py`, `hyphen/helpers.py` | Trace geometry and progress helpers |

- The CLI is `hyphenate path/to/config.yaml`, where the configuration path must
  identify the reviewed YAML file. Options layer defaults, the current generic
  document, the step document, then internal overrides. Relative paths resolve
  against the process working directory. Inspect the resolved input, output,
  plot, log and cache paths before execution. The current CLI has no dry-run mode;
  disabling fitting is not a promise of no writes. Check requested steps and
  products instead of relying only on a completion message or exit status.
- Preserve original input data, reference products, configurations and their
  environments. Treat external workspaces as read-only unless the human has
  explicitly authorized changes there. Use isolated source/environment copies
  and fresh output directories within the authorized working location; do not
  rerun an old configuration in place without checking every resolved write path.
- Preserve complete detector and cube FITS products, including WCS/ASDF metadata,
  extension identities and valid data-model handles. Do not assume the user pixel
  convention is NumPy axis order or use one indexing convention across IFU, MSA
  and merger masks without tracing the conversions. Distinguish arcseconds from
  pixels and confirm SCI/ERR/DQ/WMAP units and meaning. In particular, standard
  deviation and variance require different propagation; array position or a
  variable named `variances` does not establish the input contract. Precision
  changes, including Float32 streaming, require numerical comparison.
- Inspect side effects before importing or executing package code. Current
  orchestration sets a global NumPy seed and warning filters at import; execution
  can truncate logs, overwrite FITS, and write plots, caches and fit sidecars.
  Merger caches use output-derived names, so a reused name can reuse stale data.
  Use fresh run-specific directories and recipes; preserve completed runs.
  Saved-fit loading currently enables pickle, so load only trusted artifacts.
- For reproducible comparisons, identify the benchmark inputs, reference products
  and configuration before running. The seed is set once at import, and trace
  construction consumes the global random stream. Preserve the specified case
  order and process boundaries; do not assume cases can be reordered or split
  into fresh processes without changing results. Record worker and plotting
  settings too; current fit sidecar saving depends on plotting.
  Compare science arrays, finite/NaN masks, fitted coverage and relevant metadata,
  not regenerated gzip bytes or volatile timestamps. A merged reference must use
  the same selected inputs and settings as the candidate. Preserved outputs provide
  regression evidence alongside independent analytic/contract tests; neither replaces human
  scientific acceptance. Do not claim replay or tolerance acceptance without
  recorded evidence.
- For concurrent work, give one editor ownership of `hyphen/dewiggler_stack.py`:
  centering, batch orchestration, retries and saved-fit handling share it. Merger,
  numerical tests and documentation can proceed independently in distinct files
  and isolated environments. Do not change an editable source tree or reinstall
  dependencies while a benchmark is using it.

## Handoff for human review

- Summarize the need, changed paths, behavioral differences, exact checks and
  results, and remaining uncertainty. Identify decisions needing human judgment.
  If a check cannot run, state the concrete blocker and next check. Distinguish
  proposed, implemented, locally tested, and human-approved work. Stop at the
  reviewable patch until the comprehension checkpoint and explicit human
  acceptance are complete.
- For performance claims, report hardware, precision, workload, CPU parallelism,
  and timing boundaries. Separate cold setup from warmed execution; include
  data/WCS loading, fitting, diagnostics, and I/O for claims of overall speedup.
