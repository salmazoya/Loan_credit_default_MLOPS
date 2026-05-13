#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# jenkins-entrypoint.sh
# Fixes Docker socket permissions on every container startup, then hands off
# to the normal Jenkins entrypoint. Required on macOS Docker Desktop where
# the socket is mounted as root:root 0600.
# ─────────────────────────────────────────────────────────────────────────────

# Fix docker socket permissions as root so the jenkins user can reach the daemon
if [ -S /var/run/docker.sock ]; then
    chmod 666 /var/run/docker.sock
    echo "[entrypoint] docker.sock permissions fixed. Owner: $(stat -c '%U:%G %a' /var/run/docker.sock)"
else
    echo "[entrypoint] WARNING: /var/run/docker.sock not found — Docker builds will fail."
fi

# Drop from root to jenkins user, then start Jenkins normally
exec gosu jenkins /usr/bin/tini -- /usr/local/bin/jenkins.sh "$@"
