"""Give node access to group-restricted NVIDIA devices before dropping privileges."""

from pathlib import Path
import stat


def device_group_ids(dev: Path) -> set[int]:
    groups = set()
    for device in dev.glob("nvidia*"):
        info = device.stat()
        # Do not grant access to directories or privileged nvidia-caps devices.
        group_access = (info.st_mode >> 3) & 0o6
        other_access = info.st_mode & 0o6
        if stat.S_ISCHR(info.st_mode) and group_access & ~other_access:
            groups.add(info.st_gid)
    return groups


def grant_device_groups(group_file: Path, gids: set[int]) -> None:
    if not gids:
        return
    gids = gids.copy()
    # Edit the existing inode: startup has no CHOWN capability, and replacing
    # /etc/group would require restoring its ownership. Startup is single-threaded.
    with group_file.open("r+", encoding="utf-8") as stream:
        original = stream.read()
        entries = [line.split(":") for line in original.splitlines()]
        names = {entry[0] for entry in entries}
        for entry in entries:
            if len(entry) != 4 or int(entry[2]) not in gids:
                continue
            gids.discard(int(entry[2]))
            members = entry[3].split(",") if entry[3] else []
            if "node" not in members:
                members.append("node")
                entry[3] = ",".join(members)
        for gid in sorted(gids):
            name = f"autoresearch-gpu-{gid}"
            while name in names:
                name += "-device"
            names.add(name)
            entries.append([name, "x", str(gid), "node"])
        updated = "".join(":".join(entry) + "\n" for entry in entries)
        if updated != original:
            stream.seek(0)
            stream.write(updated)
            stream.truncate()


if __name__ == "__main__":
    grant_device_groups(Path("/etc/group"), device_group_ids(Path("/dev")))
