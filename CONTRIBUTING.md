# Contributing to microrts-drl-uecd

Solo Master's-thesis project, but it follows team-style conventions on purpose:
a clean, reviewable history is part of the deliverable.

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
3. Open a PR whose body covers:
   - **What** changed
   - **Why** it is needed
   - **Out of scope** — what the PR deliberately does not touch
   - **Test plan** — how the change was verified
4. Self-review the diff in the **Files changed** tab: scope matches the title,
   no stray files, no debug leftovers.
5. Squash & merge.

Direct pushes to `main` are blocked by branch protection.

## Merge strategy

**Squash & merge** — one PR = one commit on `main`, linear history.

After merge:

```bash
git checkout main && git pull && git branch -d <branch> && git fetch --prune
```
