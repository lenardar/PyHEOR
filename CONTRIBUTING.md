# Development and versioning

## Version rules

PyHEOR uses `major.minor.patch`. Versions identify completed, verified batches
of changes, including local development milestones. A version change does not
by itself mean that the package has been published.

| Change | While version is `0.x.y` | From `1.0.0` onward |
|---|---|---|
| Core modeling API removal or incompatible change; intentional change to units or calculation conventions | Increase minor, reset patch | Increase major, reset minor and patch |
| Small auxiliary API removal or simplification, with model calculations and core workflows unchanged | Increase patch; document the incompatibility | Increase major, reset minor and patch |
| New public functionality with existing usage preserved | Increase minor, reset patch | Increase minor, reset patch |
| Bug fixes, plotting improvements, documentation/examples, internal refactoring with existing usage preserved | Increase patch | Increase patch |

A numerical bug fix can change results while remaining a patch change. Its
CHANGELOG entry must explain which results change and why. An intentional
change in model assumptions or accounting rules belongs to the first row.

During `0.x` development, removing a narrowly scoped auxiliary plotting
interface can be a patch change. For example, removing the single-parameter
OWSA plot while retaining OWSA calculations, result tables and tornado plots
qualifies. This exception does not apply to model construction, time/reward
inputs, calculation conventions, or core result interfaces. State the removed
API explicitly in CHANGELOG; a patch number does not imply full compatibility
under this development policy. From `1.0.0`, any public API removal requires
a major version increase.

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

## Publishing to PyPI

Configure a PyPI Trusted Publisher for GitHub owner `lenardar`, repository
`PyHEOR`, workflow `publish.yml`, and environment `pypi`. For the first upload,
use a pending publisher with project name `pyheor`.

After explicitly approving publication, create a GitHub Release from the
reviewed commit with tag `v<version>` (for example, `v0.4.0`). Publishing the
Release starts `.github/workflows/publish.yml`; saving a draft does not.

The workflow checks that the tag, package metadata and source versions match,
builds the wheel and source distribution, validates their metadata and README,
and runs the tests against the installed wheel. Only then does it publish to
PyPI using Trusted Publishing and attach the same distributions to the GitHub
Release. Check the Actions run for completion before announcing availability.
