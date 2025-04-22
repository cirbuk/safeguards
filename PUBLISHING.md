# Publishing to PyPI

This project uses GitHub Actions to automatically publish releases to PyPI. The workflow is configured to trigger on pushes to the main branch and when new releases are created.

## GitHub Environment Setup

The workflow uses a GitHub environment called `pypi-publish` for secure credential management. Here's how to set it up:

1. Go to your repository on GitHub
2. Click on "Settings" > "Environments"
3. Click "New environment"
4. Name it `pypi-publish`
5. (Optional) Add environment protection rules and required reviewers
6. Click "Configure environment"

## Setting Up PyPI Trusted Publishing

This project uses PyPI's trusted publishing with OpenID Connect (OIDC), which is more secure than using API tokens.

1. Create an account on [PyPI](https://pypi.org/) if you don't have one
2. Go to your PyPI account settings > "Add a new pending publisher"
3. Fill in the form:
   - Project name: `agent-safeguards`
   - Workflow name: `Publish Python Package`
   - Environment name: `pypi-publish`
   - Repository owner: `YOUR_GITHUB_USERNAME_OR_ORG`
   - Repository name: `agent-safeguards`
4. Submit the form

## Alternative: Using PyPI API Token

If you prefer using a PyPI API token instead of trusted publishing:

1. Create an account on [PyPI](https://pypi.org/) if you don't have one
2. Go to your PyPI account settings > "API tokens"
3. Create a new API token with the scope set to the `agent-safeguards` project
4. Go to your GitHub repository settings > "Environments" > "pypi-publish" > "Add secret"
5. Add a secret named `PYPI_API_TOKEN` with the value of your PyPI token
6. Update the `.github/workflows/publish.yml` file to use the token:

```yaml
- name: Publish to PyPI
  uses: pypa/gh-action-pypi-publish@release/v1
  with:
    password: ${{ secrets.PYPI_API_TOKEN }}
```

## Versioning

The package version is defined in `pyproject.toml`. There are two ways to update the version:

1. **Automatic Version Bumping:**

   Use the "Bump Version" GitHub Action workflow:

   - Go to your repository on GitHub
   - Navigate to "Actions" > "Bump Version"
   - Click "Run workflow"
   - Select the type of version bump (patch, minor, or major)
   - Click "Run workflow" again

   This will automatically update the version in `pyproject.toml`, commit the change, create a tag, and trigger the publishing workflow.

2. **Manual Version Update:**

   Manually edit the version in `pyproject.toml` and push to main.

## When Does Publishing Happen?

The package will be published to PyPI automatically in the following cases:

1. When you push changes to the `main` branch that modify any of these files:
   - `src/**` (any file in the src directory)
   - `pyproject.toml`
   - `setup.py`
   - `setup.cfg`
   - `README.md`

2. When you create a new tag with a version number (e.g., `v0.1.0`)

3. When you create a new release through the GitHub interface

Each of these actions will trigger the CI pipeline, run tests, and if successful, publish the package to PyPI.

## Manual Publishing

If needed, you can manually publish a release by:

1. Creating a new GitHub release with a tag like `v0.1.0`
2. The workflow will automatically publish the package to PyPI
