# Independent Pi experiment image. This is NOT the main project's devcontainer.
# Customize with your scientific runtime, or replace FROM with your project's runtime image.
FROM node:24-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git python3 python3-venv python3-pip ripgrep jq bash \
    fd-find zsh iptables util-linux htop \
    && ln -s /usr/bin/fdfind /usr/local/bin/fd \
    && usermod --shell /bin/zsh node \
    && rm -rf /var/lib/apt/lists/*

# Provide uv without writing to the project's bind-mounted files.
RUN python3 -m venv /opt/uv \
    && /opt/uv/bin/pip install --no-cache-dir 'uv>=0.9,<1'
ENV PATH="/opt/uv/bin:${PATH}"

# The Pi CLI. Installed into the image as root;
# Pi runs as the unprivileged node user.
RUN npm install --global @earendil-works/pi-coding-agent

# Install the interactive shell as node so its configuration stays user-owned.
USER node
ENV HOME=/home/node SHELL=/bin/zsh
RUN git clone --depth=1 https://github.com/ohmyzsh/ohmyzsh.git /home/node/.oh-my-zsh \
    && printf '%s\n' \
        'export ZSH="$HOME/.oh-my-zsh"' \
        'ZSH_THEME="robbyrussell"' \
        "zstyle ':omz:update' mode disabled" \
        'plugins=(git)' \
        'source "$ZSH/oh-my-zsh.sh"' \
        > /home/node/.zshrc
USER root
# Root is needed only for firewall startup; do not point its shell at node's home.
ENV HOME=/root SHELL=/bin/bash

# Startup policy, node helper, and Pi defaults are immutable to the agent.
COPY --chown=0:0 init-firewall.py init-gpu.py firewall-allowlist.txt entrypoint.sh init-node.py pi-settings.json /usr/local/lib/autoresearch/
RUN chmod 0755 /usr/local/lib/autoresearch \
    && chmod 0444 /usr/local/lib/autoresearch/*

RUN mkdir -p /workspace/experiments/results /project /home/node/.pi \
    && chmod 0700 /home/node/.pi \
    && chown -R node:node /workspace /home/node/.pi

# Create correctly typed mountpoints without giving node writable placeholder
# ancestors. JSON exec form keeps host path text out of shell interpretation.

RUN ["python3", "-c", "import json\nimport os\nfrom pathlib import Path\nimport sys\n\nfor mount in json.loads(sys.argv[1]):\n    target = Path(mount[\"target\"])\n    if mount[\"directory\"]:\n        target.mkdir(parents=True, exist_ok=True)\n    else:\n        target.parent.mkdir(parents=True, exist_ok=True)\n        target.touch()\nfor root, directories, filenames in os.walk(\"/project\"):\n    os.chmod(root, 0o555)\n    for filename in filenames:\n        os.chmod(Path(root) / filename, 0o444)", "[{\"directory\": false, \"target\": \"/project/README.md\"}, {\"directory\": true, \"target\": \"/project/math\"}, {\"directory\": false, \"target\": \"/project/pyproject.toml\"}, {\"directory\": false, \"target\": \"/project/uv.lock\"}]"]

USER root
WORKDIR /workspace/experiments

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/project:/project/src \
    TRACKIO_DIR=/workspace/experiments/trackio \
    TRACKIO_STORAGE_MODE=jsonl \
    TRACKIO_SQLITE_MMAP_SIZE=0 \
    XDG_CACHE_HOME=/workspace/experiments/results/.cache \
    UV_CACHE_DIR=/workspace/experiments/results/.cache/uv \
    UV_PROJECT_ENVIRONMENT=/tmp/experiments/.venv \
    JAX_COMPILATION_CACHE_DIR=/workspace/experiments/results/.cache/jax \
    TMPDIR=/tmp

# docker exec bypasses the entrypoint and inherits the image environment.
# Keep node defaults here; the entrypoint sets root's privileged environment.
ENV HOME=/home/node SHELL=/bin/zsh USER=node LOGNAME=node

ENTRYPOINT ["/bin/bash", "-p", "/usr/local/lib/autoresearch/entrypoint.sh"]
CMD ["sleep", "infinity"]
