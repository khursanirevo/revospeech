## Summary

<!-- 1-2 sentence summary of the changes in this PR. -->

## Related issue

<!-- Link the issue this PR addresses. Use "fixes #N" to auto-close, or
     "refs #N" to reference without closing. -->

- Fixes #
- Refs #

## Type of change

<!-- Check all that apply. -->

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality
      to not work as expected)
- [ ] Refactor (code restructuring, no behavior change)
- [ ] Documentation update
- [ ] Test improvement
- [ ] Chore / CI / build configuration

## Checklist

Before requesting review, confirm the following:

- [ ] Code follows the project style (`ruff check` passes)
- [ ] Code is formatted (`ruff format --check` passes)
- [ ] Tests added for new functionality
- [ ] All existing tests pass (`uv run pytest`)
- [ ] `CHANGELOG.md` updated (if user-facing change)
- [ ] Documentation updated (`README.md`, `AGENTS.md`, `CONTRIBUTING.md` as
      needed)
- [ ] No new linting warnings introduced
- [ ] Commit messages follow
      [Conventional Commits](https://www.conventionalcommits.org/)

## Testing instructions

<!-- How can a reviewer test this change? Include commands, sample inputs,
     or steps to reproduce the expected behavior. -->

```bash
# Example:
uv run pytest tests/ -q
revospeech asr --model revovoice --input samples/test.wav
```

## Screenshots / output

<!-- For UI, CLI output, or audio-related changes, include before/after
     examples or logs that demonstrate the change works correctly. -->

## Additional notes

<!-- Any performance implications, migration steps, breaking changes,
     or context that reviewers should be aware of. -->
