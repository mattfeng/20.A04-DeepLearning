# Pi autoresearch instructions

- `/project` contains only explicitly selected read-only source mounts, not the
  whole parent repository. See the workspace README for exact target mappings;
  with no selections, `/project` is empty. Selected directories expose descendants.
- `/workspace/experiments` is a read/write mount of this entire workspace;
  changes propagate back to the host. Prefer its `results/` subdirectory for
  experiment artifacts. It is not a separate mount; there is no top-level `/results`.
- Use Pi's built-in `read`, `bash`, `edit`, and `write` tools. The user names the
  experiment and goals in chat. Inspect source and CLI help to discover commands,
  configuration, dependencies, and output conventions; do not require a particular
  filename, CLI, seed flag, or metrics format. Choose output paths explicitly;
  run directories, logs, metadata, and rankings are not generated automatically.
- Container configuration supplies shared defaults:
  `TRACKIO_DIR=/workspace/experiments/trackio`,
  `TRACKIO_STORAGE_MODE=jsonl`, `TRACKIO_SQLITE_MMAP_SIZE=0`,
  `XDG_CACHE_HOME=/workspace/experiments/results/.cache`,
  `UV_CACHE_DIR=/workspace/experiments/results/.cache/uv`,
  `UV_PROJECT_ENVIRONMENT=/tmp/experiments/.venv`,
  `JAX_COMPILATION_CACHE_DIR=/workspace/experiments/results/.cache/jax`,
  `PYTHONPATH=/project:/project/src`, and `PYTHONDONTWRITEBYTECODE=1`.
  The image sets `TMPDIR=/tmp`; these are not per-command overrides.
  Model/thinking defaults are initialized in node's global Pi settings.
- Selected source inputs cannot be edited through their read-only mounts.
  Workspace experiments and configuration are writable; change setup only when
  requested. Report results using the project's actual evidence.
- Pi and Bash commands run as `node` without sudo or firewall capabilities.
  Root startup installs an outbound allowlist and drops all capability bounding
  bits. External access is IPv4 HTTPS to startup-resolved allowed hosts and DNS
  to configured IPv4 resolvers; external IPv6 and other outbound traffic are
  blocked. Do not attempt to bypass or modify the firewall.
- If a required host is blocked, report it. The user must edit
  `.devcontainer/firewall-allowlist.txt` from the host or regular development
  container and rebuild; the active policy is a root-owned baked copy. DNS
  address rotation requires a restart. Image builds are not filtered.
- Bash inherits Pi's environment, including available credentials, and can read
  accessible credential files or Pi state. Read-only mounts do not hide secrets.
  Do not quote or transmit credentials. The IP allowlist is not domain security
  or total isolation: shared IPs can expose other services, and permitted
  services or DNS can exfiltrate data.
