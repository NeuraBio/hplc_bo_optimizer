#!/bin/bash

# Script to install Git hooks for the HPLC BO Optimizer project

# Get the absolute path to the repository root
REPO_ROOT="$(git rev-parse --show-toplevel)"
SCRIPTS_DIR="$REPO_ROOT/scripts"
HOOKS_DIR="$REPO_ROOT/.git/hooks"

echo "Installing Git hooks..."

# Create pre-commit hook that calls our docker-pre-commit.sh script
cat > "$HOOKS_DIR/pre-commit" <<EOF
#!/bin/bash
# This hook was installed by scripts/install-git-hooks.sh

# Call the docker-pre-commit.sh script with all arguments
exec "$SCRIPTS_DIR/docker-pre-commit.sh" run --all-files
EOF

# Make the hook executable
chmod +x "$HOOKS_DIR/pre-commit"

echo "Git hooks installed successfully!"
echo "The pre-commit hook will run checks inside the Docker container."
echo "Make sure the container is running before committing (make docker-up)."
