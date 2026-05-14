# Contributing to tralda

Thank you for your interest in contributing to `tralda`!
This page provides guidelines for contributing to the project, including how to set up your
development environment and submit changes.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
By participating, you are expected to uphold this code.

In short:

- Be respectful and inclusive in all interactions.
- Harassment, discrimination, and personal attacks of any kind will not be tolerated.
- Focus feedback on ideas and code, not on individuals.

If you witness or experience unacceptable behavior, please report it by opening a
[GitHub issue](https://github.com/david-schaller/tralda/issues) or by contacting the
maintainer directly.

## Reporting issues

If you encounter a bug or have a feature request, please check the
[issue tracker](https://github.com/david-schaller/tralda/issues) to see if there is an existing
issue for it.
If not, please open a new issue with a clear and descriptive title and a detailed description of
the problem or feature request.
If applicable, please include steps to reproduce the issue, expected and actual behavior, and any
relevant screenshots or error messages.

## Pull Request workflow

Please keep pull requests focused and reasonably scoped — one feature or bug fix per PR is
preferred.
Target the `develop` branch for all pull requests unless explicitly directed otherwise.

### External contributors (fork-based workflow)

If you do not have write access to the repository, please use the standard fork-based workflow:

1. [Fork](https://github.com/david-schaller/tralda/fork) the repository on GitHub.
2. Clone your fork locally and add the upstream remote:
   ```bash
   git clone https://github.com/<YOUR_USERNAME>/tralda.git
   cd tralda
   git remote add upstream https://github.com/david-schaller/tralda.git
   ```
3. Create a feature branch off `develop`:
   ```bash
   git fetch upstream
   git checkout -b my-feature upstream/develop
   ```
4. Make your changes, commit them, and push to your fork:
   ```bash
   git push origin my-feature
   ```
5. Before opening a PR, sync your branch with upstream to avoid merge conflicts:
   ```bash
   git fetch upstream
   git rebase upstream/develop
   ```
6. Open a pull request from `<YOUR_USERNAME>/tralda:my-feature` against
   `david-schaller/tralda:develop` on GitHub.

### Collaborators with repository access

If you have write access to the repository, you can work directly on a branch in the main
repository without forking:

1. Create a feature branch off `develop`:
   ```bash
   git fetch origin
   git checkout -b my-feature origin/develop
   ```
2. Make your changes, commit them, and push:
   ```bash
   git push origin my-feature
   ```
3. Open a pull request against the `develop` branch on GitHub.

### All contributors

Regardless of workflow, when opening a pull request please:

- Provide a clear title and description summarizing the changes.
- Reference any related issues (e.g., `Closes #42`).
- Request a review from a maintainer.
- Ensure branches are named descriptively (e.g., `fix/tree-construction-bug` or
  `feature/new-supertree-algorithm`).
- Ensure that commit messages are concise and ideally follow the format:
  `<type>(<scope>): <description>`, where `<type>` is one of
  `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, or `test`.

The pull request will be reviewed and you may be asked to make further changes before it is
merged.

### Changelog

Please update `CHANGELOG.md` when contributing changes.
Add a new `## [Unreleased]` section at the top of the file if one does not already exist, and
add your entry there.

Entries are organized into subsections by change type. Use the following categories as applicable:

| Category | Description |
| --- | --- |
| `🌟 Features` | New features or enhancements |
| `🚨 Breaking changes` | Changes that break backward compatibility |
| `🐛 Bug fixes` | Bug fixes |
| `⚡️ Performance` | Performance improvements |
| `♻️ Refactorings` | Internal refactorings without behavior changes |
| `🎨 Style` | Code style and formatting changes |
| `📚 Documentation` | Documentation updates |
| `📦 Build` | Build system and dependency changes |

Each entry should be a concise bullet point describing the change and, if applicable, reference
the relevant issue or pull request number.

### Checklist before submitting a Pull Request

When submitting a pull request, you agree that your contribution will be licensed under the same
license as `tralda` (MIT License).

Before submitting your pull request, make sure that:

- The `pre-commit` hooks pass (see [Code style and linting](#code-style-and-linting)).
- All tests pass locally (see [Running Tests](#running-tests)) and the
  [CI workflow](https://github.com/david-schaller/tralda/actions/workflows/ci.yml) passes.
- The documentation is updated if your changes affect the public API or add new features (see
  [Documentation](#documentation)).
- An entry has been added to the changelog (see [Changelog](#changelog)).

!!! note
    After your pull request is merged, the maintainer will include your changes in the next release.
    If you need a release sooner, feel free to open a
    [GitHub issue](https://github.com/david-schaller/tralda/issues) or contact the maintainer
    directly.

## Development environment setup with uv

If you want to contribute, please use the package and project manager
[uv](https://docs.astral.sh/uv/).
See [this page](https://docs.astral.sh/uv/getting-started/installation/) for installation
instructions.

To set up `uv`, navigate to the root directory of your local `tralda` repository (or your fork
of it — see [Pull Request workflow](#pull-request-workflow)) and create a new virtual
environment that is managed by `uv`:

```bash
cd <MY_PATH_TO>/tralda
uv sync
```

A new virtual environment will be created in the `.venv` directory of your local `tralda`
repository, and all dependencies of `tralda` will be installed in this virtual environment.
To activate this virtual environment, run the following command:

```bash
# On Linux or MacOS
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

### Managing dependencies

Dependencies are declared in `pyproject.toml`.
Runtime dependencies are listed under `[project] > dependencies`, and optional dependency groups
(e.g., `docs`) are listed under `[dependency-groups]`.

To add a new runtime dependency, use:

```bash
uv add <package>
```

To add a dependency to a specific group (e.g., `docs`), use:

```bash
uv add --group docs <package>
```

To remove a dependency, use:

```bash
uv remove <package>
```

These commands update `pyproject.toml` and `uv.lock` automatically.
Please commit both files when adding or changing dependencies.

## Code style and linting

### General guidelines

Please follow the [Google style guide](https://google.github.io/styleguide/pyguide.html) for Python
code style and documentation.
Additionally, please adhere to the following guidelines:

- Keep line lengths to a maximum of **100** characters.
- Use type hints for all function arguments and return types.
- The `Returns` section in docstring should use an additional indentation level for text that does
  not fit in a single line to ensure proper formatting in the generated documentation. In case of
  multiple return values, start the description of each return value on a new line using the same
  indentation level as the first return value.
- f-strings should be used for string formatting whenever possible.

Make sure to write clear and concise docstrings for all functions, classes, and modules.

### pre-commit

Please use [pre-commit](https://pre-commit.com) for automated code formatting and linting.
To install it and initialize it for your local `tralda` repository, follow these steps:

Install `uv` as described above.
Run the following command (after which you should be able to run `pre-commit` from anywhere):

```bash
uv tool install pre-commit --with pre-commit-uv
```

Navigate to the root directory of your local `tralda` repository and install `pre-commit`
as a git hook in the `tralda` repository:

```bash
cd <MY_PATH_TO>/tralda
pre-commit install
```

After this, `pre-commit` will automatically run the configured hooks (e.g., code formatting and
linting) before each commit, ensuring that your code adheres to the project's coding standards.

## Running Tests

The test suite uses [pytest](https://docs.pytest.org/).
Make sure you have set up your development environment with `uv` as described in the previous
section.

Install the test dependencies and run the tests:

```bash
uv sync --group test
uv run pytest
```

This will discover and run all tests in the `tests/` directory.

### Continuous Integration

The same test suite runs automatically via the
[CI GitHub Actions workflow](https://github.com/david-schaller/tralda/actions/workflows/ci.yml)
on every push and pull request, across all supported Python versions.
The workflow also runs the `pre-commit` hooks to enforce code style and linting.

You can check the current status of the CI pipeline via the badge at the top of the
[README](https://github.com/david-schaller/tralda#readme).

## Documentation

### Setting up MkDocs

The documentation of `tralda` is maintained in the `docs` folder of the repository and is built
using [MkDocs](https://www.mkdocs.org/).

To install it, run the following command:

```bash
cd <MY_PATH_TO>/tralda
uv sync --group docs

# or if you want to install all dependency groups, including the `docs` group:
uv sync --all-groups
```

After installing the dependencies, you can serve the documentation locally by running the following
command in the root directory of your local `tralda` repository:

```bash
uv run mkdocs serve
```

This will start a local development server, and you can access the documentation by navigating to
the URL displayed in the terminal.

The documentation will automatically be deployed to GitHub Pages when pushing or merging to the
`main` or `develop` branch:

| Branch | URL |
| --- | --- |
| `main` | https://david-schaller.github.io/tralda/ |
| `develop` | https://david-schaller.github.io/tralda/dev/ |

### Updating the documentation

When contributing to `tralda`, please make sure to update the documentation accordingly.

The documentation files in the `docs` folder are written in Markdown and are organized as follows:

- `index.md`: The main page of the documentation.
- `installation.md`: Instructions for installing `tralda`.
- `guide/`: A folder containing user guides and tutorials for using `tralda`.
- `api/`: A folder containing the API reference for `tralda`.
- `citation.md`: Information on how to cite `tralda` in academic publications.
- `contributing.md`: This file, containing guidelines for contributing to `tralda`.

The API reference is mostly generated from the docstrings in the code, but you may need to update
the corresponding files in the `docs/api` folder to include new modules or reflect changes in the
package structure when contributing to `tralda`.
If you add a new function or class, please add a docstring to it following the Google style guide.

The user guides and tutorials in the `docs/guide` folder are mostly written manually, so you will
need to update them manually when contributing to `tralda` if you add new features or want to
provide new guides or tutorials.

If you add or restructure documentation files, please make sure to update the `nav` section in the
`mkdocs.yml` file accordingly to include the new files and reflect the new structure.
