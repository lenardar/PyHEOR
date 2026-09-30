# Development and versioning

## Version rules

PyHEOR uses `major.minor.patch`. Versions identify completed, verified batches
of changes, including local development milestones. A version change does not
by itself mean that the package has been published.

| Change | While version is `0.x.y` | From `1.0.0` onward |
|---|---|---|
| Public API removal or incompatible change; intentional change to units or calculation conventions | Increase minor, reset patch | Increase major, reset minor and patch |
| New public functionality with existing usage preserved | Increase minor, reset patch | Increase minor, reset patch |
| Bug fixes, plotting improvements, documentation/examples, internal refactoring with existing usage preserved | Increase patch | Increase patch |

A numerical bug fix can change results while remaining a patch change. Its
CHANGELOG entry must explain which results change and why. An intentional
change in model assumptions or accounting rules belongs to the first row.

## When to update

1. Record ongoing work in the top `Unreleased` section of `CHANGELOG.md`.
2. Treat a coherent task or agreed group of tasks as one iteration. Individual
   edits, intermediate fixes, and each conversation message are not versions.
3. Once the iteration is complete and relevant verification passes, choose
   the highest applicable change level and update the version once. Mixed
   changes do not produce multiple artificial versions.
4. Move the iteration's notes into a version section with its completion date
   (`YYYY-MM-DD`). Keep `Unreleased` above completed versions. Do not rewrite
   completed version entries to include later work; correct factual errors
   explicitly when necessary.
5. A verification failure keeps the iteration in progress. If follow-up fixes
   are part of that same unfinished iteration, keep its target version. A new
   iteration after completion receives a new version.

Do not invent historical releases by splitting a batch that was never
completed separately. For example, explicit cycle/reward redesign and its
verification can form `0.2.0`; a subsequent completed plotting fix can be
`0.2.1`; a subsequent new analysis feature can be `0.3.0`.

## Completion checklist

- Synchronize `pyproject.toml`, `src/pyheor/__init__.py` (`__version__`), and
  the **PyHEOR package entry** in `uv.lock`. Do not change dependency versions
  merely to update the project version.
- Update `CHANGELOG.md` with functionality, fixes and material changes in
  results. Keep README focused on current capabilities, examples and usage;
  API migration history belongs in CHANGELOG.
- Run verification appropriate to the changes. Model calculation changes
  require relevant numerical regressions; plotting changes require data
  constraint checks and visual inspection. Documentation-only changes need
  link/content checks and execution of changed code examples when relevant.
- Reinstall the local editable package when its installed metadata is stale.
  Confirm the installed package version and `pyheor.__version__` agree.
- Report the old/new version, main changes and verification outcome.

Creating Git release tags, pushing changes or publishing a package requires
an explicit release instruction; completing a local iteration does not
authorize those actions.
