# Contributing to microrts-drl-uecd

Solo Master's-thesis project, but it follows team-style conventions on purpose:
a clean, reviewable history is part of the deliverable.

External contributions are welcome. For substantial changes (new training
features, new architectures, breaking API changes), please open an issue first
to discuss the design. It avoids wasted work on both sides. Small fixes
(typos, build issues, dead links, lint warnings) can be sent as a PR directly.

## Branch naming

Format: `<type>/<short-kebab-case>`

| Type       | Use for                                  |
|------------|------------------------------------------|
| `feat`     | New feature / capability                 |
| `fix`      | Bug fix                                  |
| `docs`     | Documentation only                       |
| `refactor` | Code restructure without behavior change |
| `test`     | Add or modify tests                      |
| `chore`    | Maintenance, deps, build/infra config    |
| `ci`       | CI/CD workflow changes                   |

Examples: `feat/env-stack`, `fix/ppo-advantage-norm`, `docs/dissertation`.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/): subject in
imperative present ("add", not "added"). The body explains the *why*; the
*what* lives in the diff.

```
<type>(<optional-scope>): <subject>

<optional body, wrapped at ~72 chars>
```

## Pull request workflow

1. Branch off `main`: `git checkout -b <type>/<name>`
2. Commit, then push: `git push -u origin <branch>`
3. Open a PR. The body should cover the four sections below; the
   [`.github/pull_request_template.md`](.github/pull_request_template.md)
   pre-fills this skeleton automatically:
   - **What** changed
   - **Why** it is needed
   - **Out of scope**: what the PR deliberately does not touch
   - **Test plan**: how the change was verified
4. Self-review the diff in the **Files changed** tab: scope matches the title,
   no stray files, no debug leftovers.
5. Squash & merge.

Direct pushes to `main` are blocked by branch protection.

## Local checks

Before pushing:

```bash
uvx ruff@0.15.14 check .      # lint
uvx ruff@0.15.14 format .     # auto-format
```

Or install the hooks once so they run on every commit:

```bash
pipx install pre-commit && pre-commit install
```

CI runs the same `ruff check` + `ruff format --check` on every PR; it must be green
before merge.

## Merge strategy

**Squash & merge**: one PR is one commit on `main`, linear history.

After merge:

```bash
git checkout main && git pull && git branch -d <branch> && git fetch --prune
```
