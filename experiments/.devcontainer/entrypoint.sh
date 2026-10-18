#!/bin/bash
set -euo pipefail

# Root startup must not use the node defaults inherited by docker exec.
export HOME=/root SHELL=/bin/bash USER=root LOGNAME=root

# Only image-owned executable directories participate in privileged startup.
export PATH=/opt/uv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
/usr/bin/python3 -I /usr/local/lib/autoresearch/init-firewall.py

# No capabilities survive into the command or any of its descendants.
exec /usr/bin/setpriv \
    --reuid=node --regid=node --init-groups \
    --bounding-set=-all --inh-caps=-all --ambient-caps=-all \
    --no-new-privs -- /usr/bin/env \
    HOME=/home/node SHELL=/bin/zsh USER=node LOGNAME=node \
    /usr/bin/python3 -I /usr/local/lib/autoresearch/init-node.py "$@"
