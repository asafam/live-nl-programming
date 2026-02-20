---
name: commit
description: Create a git commit with a concise message
disable-model-invocation: true
model: haiku
---

Create a git commit for all staged and unstaged changes. Follow these steps:

1. Run `git status` and `git diff` to understand the changes.
2. Stage the relevant changed files by name (do not use `git add -A` or `git add .`).
3. Write a concise commit message (1-2 sentences) that focuses on the "why" not the "what". End with:
   ```
   Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
   ```
4. Run `git status` after committing to verify success.

If the user provided arguments, use them as guidance for the commit message: $ARGUMENTS
