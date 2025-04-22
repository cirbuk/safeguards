# Safeguards Configuration Guide

## Table of Contents
1. [Security Rules Configuration](#security-rules-configuration)
2. [Pre-commit Hooks](#pre-commit-hooks)
3. [GitHub Actions Security Workflow](#github-actions-security-workflow)
4. [Security Scanning Tools](#security-scanning-tools)
5. [Troubleshooting](#troubleshooting)

## Security Rules Configuration

### Basic Rule Chain Setup
```python
from safeguards.rules.base import RuleChain
from safeguards.rules.defaults import PermissionGuardrail, SecurityContextRule

chain = RuleChain()
chain.add_rule(rule1)
chain.add_rule(rule2)
```

### Default Rules

#### 1. Permission Guardrail
```python
permission_rule = PermissionGuardrail(
    required_permissions={"read", "write"},
    role_permissions={
        "admin": {"read", "write", "delete"},
        "editor": {"read", "write"},
        "viewer": {"read"},
    }
)
```

#### 2. Security Context Rule
```python
security_rule = SecurityContextRule(
    required_security_level="medium",  # Options: low, medium, high
    allowed_environments={"prod", "staging", "dev"}
)
```

#### 3. Resource Limit Rule
```python
resource_rule = ResourceLimitRule(
    max_memory_mb=1024,
    max_cpu_percent=80
)
```

#### 4. Rate Limit Rule
```python
rate_rule = RateLimitRule(
    max_requests=100,
    time_window_seconds=60
)
```

### Rule Dependencies
Rules can specify dependencies that must be evaluated first:
```python
rule = CustomRule(
    dependencies=[PermissionGuardrail, SecurityContextRule]
)
```

### Rule Priority Levels
- `CRITICAL`: Must pass, blocks execution
- `HIGH`: Should pass, may block based on config
- `MEDIUM`: Warning if fails
- `LOW`: Informational only

## Pre-commit Hooks

### Installation
```bash
pip install pre-commit
pre-commit install
```

### Available Hooks
1. **GitLeaks**: Secret detection
   ```yaml
   - repo: https://github.com/zricethezav/gitleaks
     rev: v8.18.1
     hooks:
     - id: gitleaks
   ```

2. **Bandit**: Python security checks
   ```yaml
   - repo: https://github.com/PyCQA/bandit
     rev: 1.7.6
     hooks:
     - id: bandit
   ```

3. **Safety**: Dependency scanning
   ```yaml
   - repo: https://github.com/Lucas-C/pre-commit-hooks-safety
     rev: v1.3.3
     hooks:
     - id: python-safety-dependencies-check
   ```

### Custom Hook Configuration
See `.pre-commit-config.yaml` for full configuration options.

## GitHub Actions Security Workflow

### Workflow Triggers
- Push to main branch
- Pull requests
- Daily scheduled scan

### Available Scans
1. GitLeaks for secret detection
2. Safety Check for dependencies
3. Bandit for code analysis
4. Semgrep for pattern matching
5. Dependency Review
6. Snyk vulnerability scanning

### Required Secrets
- `GITHUB_TOKEN`: Automatically provided
- `SNYK_TOKEN`: Required for Snyk integration

## Security Scanning Tools

### Bandit Configuration
Configure in `.bandit.yml`:
```yaml
exclude_dirs: ['.git', 'tests', 'docs']
skips: []
level: LOW
confidence: LOW
```

### GitLeaks Configuration
Configure in `.gitleaks.toml`:
```toml
[allowlist]
paths = [
    '''.*test.*''',
    '''.*example.*''',
]

[[rules]]
id = "custom-pattern"
regex = '''pattern'''
```

### Semgrep Configuration
Configured in workflow:
```yaml
- name: Run Semgrep
  run: semgrep ci --config=auto
```

## Troubleshooting

### Common Issues

1. **Pre-commit Hook Failures**
   - Check hook configuration in `.pre-commit-config.yaml`
   - Run `pre-commit run --all-files` for details
   - Update hooks: `pre-commit autoupdate`

2. **GitHub Actions Failures**
   - Check workflow run logs
   - Verify required secrets are set
   - Check tool-specific configuration files

3. **Security Rule Violations**
   - Review violation messages
   - Check rule configuration
   - Verify input data format

### Best Practices

1. **Rule Chain Configuration**
   - Order rules by priority
   - Consider dependencies
   - Use appropriate priority levels

2. **Security Scanning**
   - Regular dependency updates
   - Monitor scan results
   - Address high-priority issues first

3. **Custom Rules**
   - Follow rule interface
   - Include comprehensive tests
   - Document requirements

### Getting Help
- Check GitHub issues
- Review documentation
- Contact maintainers
