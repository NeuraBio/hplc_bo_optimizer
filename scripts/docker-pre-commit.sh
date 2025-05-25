#!/bin/bash

# Run pre-commit (and other poetry/formatting) in the dev container
CONTAINER_NAME=hplc-bo-dev  # matches docker-compose.yml

# Detect if the container is running
if ! docker ps | grep -q $CONTAINER_NAME; then
  echo "Error: Docker container $CONTAINER_NAME is not running. Start with 'make docker-up'."
  exit 1
fi

# Pass through all arguments to poetry run pre-commit in Docker
# Run as appuser (the container's user) to avoid permission issues
docker exec $CONTAINER_NAME poetry run pre-commit "$@"

