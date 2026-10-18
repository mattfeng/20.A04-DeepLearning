# experiments

This directory has its own **separate** Pi devcontainer; existing development
container configurations remain unchanged. Open **this workspace folder**, not
its parent, in a second Visual Studio Code window. Choose **Dev Containers:
Reopen in Container**.

The image includes `fd` (Debian's `fd-find`), zsh, and Oh My Zsh. Both the `node`
account and VS Code's default terminal use zsh, with the git plugin and
`robbyrussell` theme. Image defaults provide `HOME=/home/node`, `SHELL=/bin/zsh`,
`USER=node`, and `LOGNAME=node`, including for `docker exec --user node`.
The privileged entrypoint explicitly sets root's environment for firewall startup
only. Oh My Zsh is installed/configured as node; root does not source node's shell
setup. Oh My Zsh automatic updates are disabled.

For an existing workspace, regenerate the setup from your host/development
container (review edits before using `--force`), rebuild this container, then
open a new terminal. Existing terminal sessions retain their running shell.

Regeneration removes obsolete `run-pi.sh`, `experiments/run.sh`,
`.pi/extensions/research-tools.ts`, and `.pi/extensions/experiments.ts` only
with `--force`. Review local edits before forcing removal. Symlinks at these
paths are rejected even with `--force`; remove them yourself after review.
Run `pi` directly from the container workspace directory.

Keep the final Dockerfile `USER root` for firewall startup. The entrypoint drops
to `node` before running the container command; VS Code uses `remoteUser: node`.
For direct Docker access, specify the user and shell:

```bash
docker exec --user node -it CONTAINER /bin/zsh -l
```

Docker runs the command you specify; setting `SHELL` does not replace an explicit
`bash` command. For direct root access, override the node environment explicitly:

```bash
docker exec --user root --env HOME=/root --env SHELL=/bin/bash --env USER=root --env LOGNAME=root -it CONTAINER /bin/bash -l
```

A plain `docker exec` defaults to the container's startup user (root), not VS
Code's remote user. Rebuild/recreate to remove completion caches left by shells
using the wrong identity; only Pi state, not the whole home directory, is
persisted by this setup. Keep Oh My Zsh's ownership checks enabled.

### Node Pi settings

After root installs the firewall, the entrypoint uses `setpriv` to drop to node
with all capabilities removed. It runs `.devcontainer/init-node.py` from the
immutable root-owned copy in `/usr/local/lib/autoresearch/`, then executes the
container command. The initializer seeds node's global
`~/.pi/agent/settings.json` (`/home/node/.pi/agent/settings.json`) from the baked
`.devcontainer/pi-settings.json`. Root's Pi settings are untouched.

A new node user receives provider `openai-codex`, model `gpt-6.1-sol`, and
`medium` thinking. Existing settings keys and `auth.json` are preserved.
The provider/model pair is seeded only when neither key is present; thinking is
filled only when missing. Malformed existing settings stop startup visibly
instead of being overwritten. The `/home/node/.pi` named volume persists
settings, authentication, and sessions across rebuilds.

Edit `.devcontainer/pi-settings.json` on the host/development container and
rebuild to change the baked seed. Startup does not overwrite existing user
choices. To change active choices, edit `~/.pi/agent/settings.json` or use Pi's
`/settings` **inside a node shell**. Model defaults do not require a project
extension or project settings.

There is no generated project `.pi/settings.json`. Regeneration removes the old
file automatically only if it matches the old generated defaults. If modified,
review it and migrate desired choices to node's settings before using `--force`
to remove it.

## Filesystem contract

| In Pi container | Access | Host content |
|---|---|---|
| Selected targets under `/project` | read-only | explicit selections listed below |
| `/workspace/experiments` | read/write bind mount | this entire folder (configuration, experiments, and results) |
| `/workspace/experiments/results` | writable subdirectory, not a separate mount | `experiments/results` |
| `/home/node/.pi` | read/write | persistent named volume for agent state |

`TRACKIO_DIR` defaults to `/workspace/experiments/trackio` for container
sessions and Pi-launched experiments. If Trackio is installed, local project
data is saved in the workspace's `trackio/` subdirectory, shared across runs.
The dashboard uses the same default. An explicit `TRACKIO_DIR` override is
preserved; Trackio is not installed by this generator.


| Resolved host source | Relative to this workspace | Container target |
|---|---|---|

| `/Users/mattfeng/20.A04-DeepLearning/README.md` | `../README.md` | `/project/README.md` |

| `/Users/mattfeng/20.A04-DeepLearning/math` | `../math` | `/project/math` |

| `/Users/mattfeng/20.A04-DeepLearning/pyproject.toml` | `../pyproject.toml` | `/project/pyproject.toml` |

| `/Users/mattfeng/20.A04-DeepLearning/uv.lock` | `../uv.lock` | `/project/uv.lock` |



Only selected sources are exposed under `/project`. Symlinks resolve to canonical
sources; a selected directory exposes all descendants. Selecting sources outside
the workspace's parent can shift the common ancestor used for target paths.
The root-owned `/project` scaffolding is not a writable work area.

Selections are saved as portable workspace-relative paths in `.autoresearch.json`.
Rerun `configure-autoresearch --workspace /path/to/this-folder` on the host to
browse from this workspace with saved choices prechecked. Use `ls`, `up`,
`cd NAME/INDEX/PATH`, `toggle INDEX/PATH`, `select .`, and `selected`; review
the printed mapping, then `done` to confirm or `cancel` without writes.
For a nonexistent or empty workspace, the browser starts in its parent instead.
Existing autoresearch workspaces start inside themselves.

Enter `1,3,5-8` directly to toggle several entries, or use `toggle 1,3,5-8`.
Ranges include both endpoints. `select 1,3,5-8` adds entries without deselecting
existing choices. Number lists must contain no spaces; repeats and overlaps
count once, and invalid batches do not change any selections.

Alternatively, pass repeatable `--read-only PATH` choices (relative to the
workspace, overriding saved choices), or `--non-interactive` to reuse saved
configuration. Without explicit choices, non-interactive setup requires saved
configuration. No writes occur until successful generation; `--dry-run` never
writes files or settings. Changed generated files require `--force` to replace.
The obsolete runner and extension files listed above require reviewed `--force`
removal; commands now run through Pi's built-in Bash tool.

The supplied workspace must be new, empty, or an existing autoresearch directory.
Setup checks this before opening the picker and again before generating files.
Non-empty unrelated directories are rejected even with `--force` or `--dry-run`;
`--force` only permits replacing generated files in an accepted workspace.

For example, initialize `--workspace /host/project/math-experiments`; the browser
starts in `/host/project`. Use `toggle math` and `toggle pyproject.toml`
(optionally `toggle uv.lock`).
After `done`, `/host/project/math` maps to `/project/math` and
`/host/project/pyproject.toml` to `/project/pyproject.toml`, without exposing
unrelated files in `/host/project`.

**Important:** the entire workspace at `/workspace/experiments` is
writable, and changes propagate back to the host, including changes to experiment
scripts and configuration. Prefer its normal `results/` subdirectory for outputs
and treat those outputs as untrusted. There is no top-level `/results`
directory or mount. Only selected source inputs under `/project` are read-only;
a broad source selection may expose a read-only alias of the workspace without
preventing writes through its workspace path. No container is a security boundary
against its host's kernel or a privileged Docker socket; no Docker socket is mounted.
Only root startup receives the capabilities needed to install the firewall and
drop to `node`; Pi and experiments run without capabilities or sudo.
The container allocates 1 GiB to `/dev/shm` via `--shm-size=1g`.

## First run

1. Open this folder in its own devcontainer window and build the container.
2. In the new container, verify the **actual selected mounts**, not just an
   unmounted placeholder directory. This checks read-only filesystem flags
   without modifying source files, and creates temporary probes to check writes:

   ```bash
   python3 - <<'PY'
   import os
   import tempfile
   paths = [
   
       "/project/README.md",
   
       "/project/math",
   
       "/project/pyproject.toml",
   
       "/project/uv.lock",
   
   ]
   for path in paths:
       assert os.statvfs(path).f_flag & os.ST_RDONLY, f"UNSAFE: {path} is not read-only"
       print("read-only:", path)
   workspace = "/workspace/experiments"
   results = os.path.join(workspace, "results")
   for path in (workspace, results):
       assert not os.statvfs(path).f_flag & os.ST_RDONLY, f"{path} is read-only"
       with tempfile.TemporaryFile(dir=path) as probe:
           probe.write(b"writable\n")
           probe.flush()
       print("writable:", path)
   assert not os.path.lexists("/results"), "unexpected top-level /results"
   PY
   ```

3. Authenticate Pi with OpenAI Codex **as node**, then launch Pi.
   The node settings described above default to `openai-codex/gpt-6.1-sol`
   with `medium` thinking. Your installed Pi/provider catalog and account must
   support this model; seeding defaults does not add it to the catalog.

   ```bash
   cd /workspace/experiments
   pi
   ```

4. Tell Pi the experiment file and goals in chat: “Use `/project/experiment.py`;
   inspect its source and `--help`, discover its configuration and output
   conventions, and run a small experiment.” Pi uses the built-in `read`,
   `bash`, `edit`, and `write` tools.

   `/project/experiment.py` is illustrative: use a file beneath one of the
   selected targets above (for example `/project/math/experiment.py`).
   No prescribed filename, CLI, or metrics format is required.

Run `pi` directly from `/workspace/experiments` with its built-in tools.
Bash executes ordinary commands with node's permissions and inherits Pi's
environment, including any credentials available there.

The Dockerfile and devcontainer configuration supply shared defaults:

| Variable | Default |
|---|---|
| `TRACKIO_DIR` | `/workspace/experiments/trackio` |
| `TRACKIO_STORAGE_MODE` | `jsonl` |
| `TRACKIO_SQLITE_MMAP_SIZE` | `0` |
| `XDG_CACHE_HOME` | `/workspace/experiments/results/.cache` |
| `UV_CACHE_DIR` | `/workspace/experiments/results/.cache/uv` |
| `UV_PROJECT_ENVIRONMENT` | `/tmp/experiments/.venv` |
| `JAX_COMPILATION_CACHE_DIR` | `/workspace/experiments/results/.cache/jax` |
| `PYTHONPATH` | `/project:/project/src` |
| `PYTHONDONTWRITEBYTECODE` | `1` |
| `TMPDIR` | `/tmp` (image default) |

These are container defaults, not per-command overrides. Prefer `results/`
for experiment outputs, choosing paths and recording evidence as appropriate
for the project. Pi does not automatically create run directories, capture
experiment logs or metadata, or rank metrics.

## Iteration

Edit project experiments from your **regular development devcontainer** and
name them in chat; no prescribed filename or CLI is required. This experiment
container cannot edit selected source mounts, but it can edit its writable
workspace setup. Rebuild the container when you change its Dockerfile.

`uv run` and `uv sync` create/use `/tmp/experiments/.venv` by default,
not `/project/.venv`. This environment is temporary and can be recreated after
container replacement or `/tmp` cleanup. For standalone creation, use
`uv venv "$UV_PROJECT_ENVIRONMENT"`; plain `uv venv` does not use that variable.
`UV_CACHE_DIR` remains under `/workspace/experiments/results/.cache/uv`.
To match complex scientific dependencies, modify the Pi
Dockerfile or base it on the same runtime image as your development container.
GPU access is not enabled; configure it deliberately if required.

## Outbound firewall

The firewall applies to Pi and every experiment command by default. Root startup
installs fail-closed IPv4/IPv6 filter rules before resolving allowlisted hosts,
then uses `setpriv` to run the node settings initializer and command as `node`
with all capability bounding bits dropped. Docker drops all capabilities except
startup-only `NET_ADMIN`, `SETUID`, `SETGID`, and `SETPCAP`; no-new-privileges
and the PID limit remain.
The agent cannot modify the root-owned firewall or obtain firewall privileges.
Configuration or DNS resolution failure stops startup with restrictive rules
remaining in place.

Allowed traffic is loopback, inbound responses to permitted connections, DNS
(TCP/UDP port 53) to IPv4 nameservers from `/etc/resolv.conf`, and IPv4 TCP
port 443 to A records resolved at startup for allowlisted hosts. Other outbound
traffic is rejected, including external IPv6. Docker NAT is preserved. There
are no SSH, host-gateway, broad GitHub CIDR, or blanket telemetry exceptions.

The defaults cover Pi/model/auth (`api.anthropic.com`, `api.openai.com`,
`auth.openai.com`, `chatgpt.com`, `pi.dev`), package downloads
(`registry.npmjs.org`, `pypi.org`, `files.pythonhosted.org`), GitHub
(`github.com`, `api.github.com`, `raw.githubusercontent.com`,
`objects.githubusercontent.com`), and editor services
(`marketplace.visualstudio.com`, `vscode.blob.core.windows.net`,
`update.code.visualstudio.com`).

For a different provider or dataset host, edit
`.devcontainer/firewall-allowlist.txt` **on the host or in your regular
development devcontainer**, then rebuild this Pi container. Use one hostname
per line; comments and blank lines are accepted. The Dockerfile copies this
allowlist, `.devcontainer/init-firewall.py`, `.devcontainer/entrypoint.sh`,
`.devcontainer/init-node.py`, and `.devcontainer/pi-settings.json` into root-owned
`/usr/local/lib/autoresearch/`. Changing a mounted copy does not alter the active
policy or settings seed; the agent cannot edit the baked files.

This is an IP allowlist, not domain-level security or total network isolation.
Shared IPs may expose other services; permitted services and DNS can be used
for data exfiltration. External access is IPv4 HTTPS only. DNS address rotation
requires restarting the container to resolve again. Image builds are not
filtered by the runtime firewall. Commands still can read credential files.

Manual checks inside this container, using `curl` if available:

```bash
curl --noproxy '*' --connect-timeout 10 -I https://api.openai.com
curl --noproxy '*' --connect-timeout 10 -I https://example.com
curl --noproxy '*' --connect-timeout 10 -6 -I 'https://[2606:4700:4700::1111]'
```

The first should return an HTTP status (any status proves reachability; do not
use `-f`). The other two should fail to connect. These checks do not establish
domain security or prevent exfiltration through allowed destinations.

## Trust and limitations

- Pi's built-in Bash tool provides general shell execution with the container
  user's permissions and the same outbound firewall as Pi.
- Bash inherits Pi's environment, including available credentials. Commands can
  also read accessible credential files, including Pi state. There is no
  environment filtering or credential isolation.
- Read-only selected source mounts prevent edits through those mounts, but do
  not redact sensitive files; broadly selected directories expose descendants.
- The image uses a conventional writable container layer for system temporary
  files. The entire workspace is writable back to the project host.
- Generated file updates are intentional: regenerate with `--force` only after
  reviewing changes; this replaces generated files.

Reference: https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/cli.md
