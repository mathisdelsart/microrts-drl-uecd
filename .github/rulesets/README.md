# Rulesets (snapshot, not auto-applied)

GitHub does not yet support declarative rulesets from a repository file
(unlike `dependabot.yml` or workflow files). The JSON in this folder is a
**read-only snapshot** committed for traceability and review; editing it
does not change branch protection. The live ruleset is configured under
**Settings → Rules → Rulesets** and via the REST API.

## Files

- `protect-main.json`: snapshot of the `protect-main` ruleset, which
  governs `main`. PRs required, squash-only merges, three required
  status checks (`ruff (lint + format)`, `java bridge (build)`,
  `pytest (smoke)`), no force-push, no deletion. Volatile fields
  (`_links`, `node_id`, `updated_at`, `current_user_can_bypass`)
  are stripped so the diff stays meaningful across re-exports.

## Refreshing the snapshot

```bash
gh api repos/mathisdelsart/microrts-drl-uecd/rulesets/16981938 \
  | python3 -c "import json,sys; d=json.load(sys.stdin); \
      [d.pop(k,None) for k in ('_links','node_id','updated_at','current_user_can_bypass')]; \
      print(json.dumps(d, indent=4))" \
  > .github/rulesets/protect-main.json
```

Run this after any change made through Settings or the API. Compare the
diff against the previous snapshot before committing; an unexpected diff
means someone changed the live ruleset out-of-band.
