# `.github/` — repository metadata

This directory holds files that GitHub interprets automatically: CI/CD pipelines,
dependency policy, security policy, issue & PR templates. None of it ships in
the Python package; it only affects how the repo behaves on GitHub itself.

## Tree

```
.github/
├── README.md                       # this file
├── SECURITY.md                     # security disclosure policy
├── dependabot.yml                  # Dependabot version-update config
├── pull_request_template.md        # auto-filled when opening a PR
├── ISSUE_TEMPLATE/                 # forms shown under "New Issue"
│   ├── bug_report.yml
│   ├── feature_request.yml
│   └── config.yml                  # disables blank issues + contact link
└── workflows/                      # GitHub Actions (CI/CD)
    ├── ci.yml                      # ruff + bridge build + pytest matrix
    ├── codeql.yml                  # static security analysis
    └── dependency-review.yml       # PR-time CVE check on dep changes
```

## File-by-file

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
- **`docker`** (monthly): bumps the base image tags in `Dockerfile` and
  `Dockerfile.gpu` (`python:3.10-slim`, `nvidia/cuda:12.4.1-...`) so OS-level
  CVEs in the underlying image get caught.

**What this file does NOT cover** (and shouldn't):
- **Security alerts on pip deps** — handled by the repo-level Settings → "Code
  security" toggles (Dependabot alerts + Dependabot security updates), enabled
  on 2026-05-30. PR #81 (Pillow 11.3.0 → 12.2.0) was opened by this channel
  after PR #80 shipped `requirements-lock.txt`.
- **Pip version bumps** — intentionally off. The repo is a thesis-companion
  artefact in a mostly-frozen state; a continuous stream of `pip` PRs would
  be noise. CVEs still flow through the security-alerts channel above.

### `pull_request_template.md`
Pre-fills the PR description with What / Why / Out of scope / Test plan.
A hidden comment at the top reminds contributors of:
- Conventional Commit title (`feat:` / `fix:` / `docs:` / `chore:` / `ci:` /
  `refactor:`)
- Branch prefix matching the title
- Squash-merge only
- Ruff & pytest must pass

The full contributor guide lives in `../CONTRIBUTING.md`.

### `ISSUE_TEMPLATE/`
Three files:
- **`bug_report.yml`** — structured form (Summary / Reproduction / Expected /
  Actual / Environment / Commit). Auto-labels with `bug`.
- **`feature_request.yml`** — structured form (Problem / Proposal /
  Alternatives / Scope). Auto-labels with `enhancement`.
- **`config.yml`** — meta-config:
  - `blank_issues_enabled: false` — forces use of the templates above; no
    free-form issues.
  - `contact_links:` — adds a "Question about the dissertation or paper"
    button that opens a mailto to the author instead of creating an issue
    (research questions don't belong in the bug tracker).

### `workflows/` — GitHub Actions

#### `ci.yml`
Runs on every push to `main` and every PR. Four jobs:

| Job (`name:` in checks) | What it does                                  | Typical duration |
| ----------------------- | --------------------------------------------- | ---------------- |
| `ruff (lint + format)`  | `ruff check` + `ruff format --check` (3.12)   | ~1 min           |
| `java bridge (build)`   | Recompile `microrts/src/` → `bridge.jar`      | ~1 min           |
| `pytest matrix (3.10/3.11/3.12)` | 3 parallel cells running the 115 smoke tests with CPU-only torch | ~3-4 min/cell |
| `pytest (smoke)`        | Empty aggregator — `needs:` the matrix above  | <1 s             |

The aggregator pattern keeps a stable required-check name on the branch-
protection rule across matrix expansion. `concurrency:` cancels superseded
PR runs but never cancels push-to-main runs.

#### `codeql.yml`
Static security + quality analysis on the Python source. Runs on push/PR to
`main` plus a weekly cron (`21 7 * * 1` = Monday 07:21 UTC) so advisories
that land between PRs still get caught. Uses the `security-and-quality`
query suite (security + broader code-quality, appropriate for a research
artefact that gets occasional contributors).

#### `dependency-review.yml`
PR-time check: diffs the PR's dependency manifests against the GitHub
Advisory Database and fails the check if the PR adds a dep with a known
**HIGH or CRITICAL** CVE. Moderate findings are reported but don't fail
(signal:noise). Complements `codeql.yml` (source-code analysis) and the
Dependabot security alerts (post-hoc bump PRs).

## Branch protection (configured outside this folder)

The `protect-main` ruleset (Settings → Rules → Rulesets) requires the
following checks to pass before a PR can merge to `main`:
- `ruff (lint + format)`
- `java bridge (build)`
- `pytest (smoke)` (aggregator)

It also enforces: PR required, squash-only, no force-push, no deletion. The
ruleset is configured via the GitHub API / UI; it is intentionally **not**
committed as a file in this repo (no GitHub-native YAML for this exists
outside the Settings UI).

## What is deliberately NOT here

- **`FUNDING.yml`** — no donation button; not relevant for a thesis artefact.
- **`CODEOWNERS`** — single-author project; auto-assigning reviewers is
  meaningless.
- **A release workflow** — releases are tagged manually when needed; no
  artefact build / publish step is automated.
- **`labeler.yml`** / auto-labelling — issue volume doesn't justify it.

Anything beyond what's listed in the tree above would be over-engineering for
the lifecycle this repo is in.
