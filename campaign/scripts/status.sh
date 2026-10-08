#!/usr/bin/env bash
# Progress of a campaign.  Usage: scripts/status.sh [--profile smoke|pilot]
set -euo pipefail
source "$(dirname "$0")/_env.sh"
xc status "$@"
