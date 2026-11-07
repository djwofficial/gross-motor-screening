# Repo admin steps (Daryn)

Status 2026-11-08: repo pushed (private). Three steps are still open; each needs Daryn's account.

## 1. Turn on CI (GitHub Actions)
The CI file `.github/workflows/ci.yml` is ready but not pushed: the GitHub CLI token has no
`workflow` scope. Run once in a terminal (opens the browser to approve):
```bash
gh auth refresh -h github.com -s workflow
```
Then commit and push `.github/workflows/ci.yml`. CI runs `ruff format --check`, `ruff check`
and `pytest` on every push to main and every pull request.

## 2. Invite the team and fill CODEOWNERS
GitHub → repo → Settings → Collaborators → add Marty, Revallo, Louise, Freya, Theresia.
Then replace the `*_USERNAME` placeholders in `.github/CODEOWNERS` and remove the leading `# `.

## 3. Lock main (needs GitHub Pro for a private repo)
`Upgrade to GitHub Pro or make this repository public` (HTTP 403 on 2026-11-08). Students get
Pro free through the GitHub Student Developer Pack (education.github.com/pack). After that:
```bash
echo '{"required_status_checks":{"strict":true,"contexts":["test"]},"enforce_admins":false,"required_pull_request_reviews":{"require_code_owner_reviews":true,"required_approving_review_count":1},"restrictions":null}' \
  | gh api -X PUT repos/djwofficial/gross-motor-screening/branches/main/protection --input -
```
This makes every change go through a PR with green CI and a review by the folder's owner.
Until then, the rule in AGENTS.md (one task = one branch = one PR) is followed by agreement.
