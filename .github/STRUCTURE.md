# `.github/` (repository metadata)

This directory holds files that GitHub interprets automatically: CI/CD pipelines,
dependency policy, security policy, issue & PR templates, branch-protection
snapshot. None of it ships in the Python package; it only affects how the repo
behaves on GitHub itself.

## Tree

```
.github/
├── STRUCTURE.md                    # this file
├── CONTRIBUTING.md                 # branch naming, Conventional Commits, PR workflow
├── CODE_OF_CONDUCT.md              # Contributor Covenant 2.1
├── SECURITY.md                     # security disclosure policy
├── dependabot.yml                  # Dependabot version-update config
├── pull_request_template.md        # auto-filled when opening a PR
├── protect-main-ruleset.json       # JSON snapshot of the live branch-protection ruleset (doc only)
├── ISSUE_TEMPLATE/                 # forms shown under "New Issue"
│   ├── bug_report.yml
│   ├── feature_request.yml
│   └── config.yml                  # disables blank issues + contact links
└── workflows/                      # GitHub Actions
    ├── ci.yml                      # ruff + pre-commit + bridge build + pytest matrix
    ├── codeql.yml                  # static security analysis (Python + Java)
    └── dependency-review.yml       # PR-time CVE check on dep changes
```

## File-by-file

### `CONTRIBUTING.md`
Contributor guide: branch naming convention (`<type>/<short-name>`),
Conventional Commits, PR workflow with the four-section template body, and
the squash-merge rule. GitHub recognises this file in `.github/`, the repo
root, or `docs/`; keeping it under `.github/` declutters the root and groups
it with the rest of the repo-metadata files.

### `CODE_OF_CONDUCT.md`
Contributor Covenant 2.1 (adapted). Same placement rationale as
`CONTRIBUTING.md`: GitHub picks it up from `.github/` just as well as from
the repo root.

### `SECURITY.md`
Disclosure policy. Says: open a public GitHub issue for security-relevant
findings (the project is academic, single-author, no production deployment, so
there is no private channel). Sensitive cases can email the author directly.
GitHub recognises this file in three locations (repo root, `.github/`, `docs/`);
keeping it under `.github/` declutters the root.

### `dependabot.yml`
Tells Dependabot to open *version-update* PRs on a schedule for:
- **`github-actions`** (monthly): bumps `actions/checkout@vN`,
  `actions/setup-python@vN`, etc. used in the workflows.

**What this file does NOT cover** (and shouldn't):
- **Security alerts on pip deps**: handled by the repo-level Settings -> "Code
  security" toggles (Dependabot alerts + Dependabot security updates), enabled
  on 2026-05-30. PR #81 (Pillow 11.3.0 -> 12.2.0) was opened by this channel
  after PR #80 shipped `requirements-lock.txt`.
- **Pip version bumps**: intentionally off. The repo is a thesis-companion
  artefact in a mostly-frozen state; a continuous stream of `pip` PRs would
  be noise. CVEs still flow through the security-alerts channel above.

### `pull_request_template.md`
Pre-fills the PR description with What / Why / Out of scope / Test plan.
A hidden comment at the top reminds contributors of:
- Conventional Commit title (`feat:` / `fix:` / `docs:` / `chore:` / `ci:` /
  `refactor:`)
- Branch prefix matching the title
- Squash-merge only
- `pre-commit run --all-files` clean (no `# noqa`) and `pytest tests/` green

The full contributor guide lives in [`CONTRIBUTING.md`](CONTRIBUTING.md).

### `ISSUE_TEMPLATE/`
Three files. GitHub auto-discovers them from this exact path; the folder
name `ISSUE_TEMPLATE/` and the `.yml` extension are hard-coded conventions.
- **`bug_report.yml`**: structured form (Summary / Reproduction / Expected /
  Actual / Environment / Commit). Auto-labels with `bug`.
- **`feature_request.yml`**: structured form (Problem / Proposal /
  Alternatives / Scope). Auto-labels with `enhancement`.
- **`config.yml`**: meta-config:
  - `blank_issues_enabled: false`: forces use of the templates above; no
    free-form issues.
  - `contact_links:` two buttons surfaced under "New Issue":
    - **Discussions**: open-ended Q&A and how-to threads (publicly
      searchable for the next person with the same question).
    - **Email the author**: research-specific questions about the
      dissertation or paper, kept out of the bug tracker.

### `protect-main-ruleset.json`
Read-only snapshot of the live `protect-main` ruleset (configured under
Settings -> Rules -> Rulesets). Committed for traceability and diff review;
editing the file does NOT change branch protection. The ruleset enforces:
- PRs required, squash-merge only, no force-push, no deletion.
- Seven required status checks (see Branch protection below).

To refresh the snapshot after any change made through Settings or the API:

```bash
gh api repos/mathisdelsart/microrts-drl-uecd/rulesets/16981938 \
  | python3 -c "import json,sys; d=json.load(sys.stdin); \
      [d.pop(k,None) for k in ('_links','node_id','updated_at','current_user_can_bypass')]; \
      print(json.dumps(d, indent=4))" \
  > .github/protect-main-ruleset.json
```

Compare the diff against the previous snapshot before committing; an
unexpected diff means someone changed the live ruleset out-of-band.

### `workflows/` (GitHub Actions)

#### `ci.yml`
Runs on every push to `main` and every PR. Five jobs:

| Job (`name:` in checks) | What it does                                  | Typical duration |
| ----------------------- | --------------------------------------------- | ---------------- |
| `ruff (lint + format)`  | `ruff check` + `ruff format --check` (3.12)   | ~1 min           |
| `pre-commit (all hooks)`| Runs every hook in `.pre-commit-config.yaml` on the full tree | ~1-2 min |
| `java bridge (build)`   | Recompile `microrts/src/` -> `bridge.jar`     | ~1 min           |
| `pytest matrix (3.10/3.11/3.12)` | 3 parallel cells running the 115 smoke tests with CPU-only torch | ~3-4 min/cell |
| `pytest (smoke)`        | Empty aggregator; `needs:` the matrix above   | <1 s             |

The aggregator pattern keeps a stable required-check name on the branch-
protection rule across matrix expansion. `concurrency:` cancels superseded
PR runs but never cancels push-to-main runs.

#### `codeql.yml`
Static security + quality analysis. Runs on push/PR to `main` plus a weekly
cron (`21 7 * * 1` = Monday 07:21 UTC) so advisories that land between PRs
still get caught. Uses the `security-and-quality` query suite (security +
broader code-quality, appropriate for a research artefact). Matrix on
`language: [python, java]`: Python covers `microrts_agent/`, Java covers the
hand-written JNI bridge under `microrts_agent/microrts/src/`. The vendored
MicroRTS engine + competition bots are excluded via `linguist-vendored` in
`.gitattributes`, so CodeQL only analyzes our own sources.

#### `dependency-review.yml`
PR-time check: diffs the PR's dependency manifests against the GitHub
Advisory Database and fails the check if the PR adds a dep with a known
**HIGH or CRITICAL** CVE. Moderate findings are reported but don't fail
(signal:noise). Complements `codeql.yml` (source-code analysis) and the
Dependabot security alerts (post-hoc bump PRs).

## Branch protection (configured outside this folder)

The `protect-main` ruleset (Settings -> Rules -> Rulesets) requires every
check below to pass before a PR can merge to `main`:
- `ruff (lint + format)`
- `pre-commit (all hooks)`
- `java bridge (build)`
- `pytest (smoke)` (aggregator)
- `Analyze (python)` (CodeQL)
- `Analyze (java)` (CodeQL)
- `Review dependency changes`

It also enforces: PR required, squash-only, no force-push, no deletion. The
ruleset is configured via the GitHub API / UI; GitHub has no native YAML
auto-import for rulesets, so [`protect-main-ruleset.json`](protect-main-ruleset.json)
in this folder is a stripped JSON snapshot kept for traceability only
(editing the file does nothing).

## What is deliberately NOT here

- **`FUNDING.yml`**: no donation button; not relevant for a thesis artefact.
- **`CODEOWNERS`**: single-author project; auto-assigning reviewers is
  meaningless.
- **A release workflow**: releases are tagged manually when needed; no
  artefact build / publish step is automated.
- **`labeler.yml`** / auto-labelling: issue volume doesn't justify it.

Anything beyond what's listed in the tree above would be over-engineering for
the lifecycle this repo is in.
