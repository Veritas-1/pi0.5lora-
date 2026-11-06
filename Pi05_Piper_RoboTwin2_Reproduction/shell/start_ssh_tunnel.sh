#!/usr/bin/env bash
# Local Ubuntu: keep this foreground process running while evaluating.
set -euo pipefail
exec ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -N -L 127.0.0.1:16000:127.0.0.1:6000 \
  -p 30185 root@sh03-ssh.gpuhome.cc
