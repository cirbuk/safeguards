# Safeguards Framework Documentation

This directory contains the configuration for the Safeguards Framework documentation site.

## Local Development

To run the documentation site locally:

1. Install the required dependencies:
   ```bash
   pip install mkdocs mkdocs-material mkdocstrings mkdocstrings-python
   ```

2. Start the development server:
   ```bash
   cd docs-site
   mkdocs serve
   ```

3. Open your browser and visit http://127.0.0.1:8000/

## Structure

- `mkdocs.yml` - Configuration file for the documentation site
- The actual documentation content is in the `../docs` directory

## Deployment

The documentation is automatically deployed to GitHub Pages when changes are pushed to the main branch.

The site will be available at: https://cirbuk.github.io/safeguards/

To manually deploy the documentation:

```bash
cd docs-site
mkdocs gh-deploy --force
```

## Customization

To customize the site, edit the `mkdocs.yml` file. Refer to the [MkDocs documentation](https://www.mkdocs.org/) and [Material for MkDocs documentation](https://squidfunk.github.io/mkdocs-material/) for more information.
