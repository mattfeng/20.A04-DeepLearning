"""Install a fail-closed, startup-resolved outbound allowlist for the container."""

import ipaddress
from pathlib import Path
import re
import socket
import subprocess


ALLOWLIST_PATH = Path("/usr/local/lib/autoresearch/firewall-allowlist.txt")
RESOLV_CONF_PATH = Path("/etc/resolv.conf")
IPTABLES_RESTORE = "/usr/sbin/iptables-restore"
IP6TABLES_RESTORE = "/usr/sbin/ip6tables-restore"
_HOST_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", re.ASCII)


def parse_allowlist(text: str) -> list[str]:
    """Accept comments and one DNS hostname per line, never addresses or URLs."""
    hosts = []
    for number, line in enumerate(text.splitlines(), start=1):
        hostname = line.partition("#")[0].strip().lower()
        if not hostname:
            continue
        labels = hostname.split(".")
        if (
            len(hostname) > 253
            or len(labels) < 2
            or not all(_HOST_LABEL.fullmatch(label) for label in labels)
        ):
            raise ValueError(f"Invalid allowlist hostname on line {number}: {hostname!r}")
        try:
            ipaddress.ip_address(hostname)
        except ValueError:
            pass
        else:
            raise ValueError(f"Allowlist entries must be hostnames, not addresses: {hostname!r}")
        # libc also accepts legacy address forms (for example 127.1 or
        # 0x7f000001); those are addresses, not DNS allowlist hostnames.
        try:
            socket.inet_aton(hostname)
        except OSError:
            pass
        else:
            raise ValueError(f"Allowlist entries must be hostnames, not addresses: {hostname!r}")
        if hostname not in hosts:
            hosts.append(hostname)
    if not hosts:
        raise ValueError("The firewall allowlist must contain at least one hostname")
    return hosts


def parse_nameservers(text: str) -> list[str]:
    """Read literal IPv4 DNS servers from resolv.conf; IPv6 stays blocked."""
    servers = set()
    for line in text.splitlines():
        fields = line.partition("#")[0].partition(";")[0].split()
        if not fields or fields[0] != "nameserver":
            continue
        if len(fields) != 2:
            raise ValueError("Invalid nameserver entry in /etc/resolv.conf")
        try:
            address = ipaddress.ip_address(fields[1])
        except ValueError as error:
            raise ValueError("Nameserver entries must be literal IP addresses") from error
        if address.version == 4:
            servers.add(str(address))
    if not servers:
        raise ValueError("The firewall requires at least one IPv4 DNS nameserver")
    return sorted(servers)


def render_ipv4_policy(nameservers: list[str], allowed_addresses: list[str]) -> str:
    """Render only the filter table, leaving Docker's NAT and mangle tables intact."""
    servers = sorted({str(ipaddress.IPv4Address(value)) for value in nameservers})
    addresses = sorted({str(ipaddress.IPv4Address(value)) for value in allowed_addresses})
    rules = [
        "*filter",
        ":INPUT DROP [0:0]",
        ":FORWARD DROP [0:0]",
        ":OUTPUT DROP [0:0]",
        "-A INPUT -i lo -j ACCEPT",
        "-A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT",
        "-A OUTPUT -o lo -j ACCEPT",
    ]
    for server in servers:
        for protocol in ("udp", "tcp"):
            rules.append(f"-A OUTPUT -d {server}/32 -p {protocol} --dport 53 -j ACCEPT")
    for address in addresses:
        rules.append(f"-A OUTPUT -d {address}/32 -p tcp --dport 443 -j ACCEPT")
    # Never accept arbitrary established outbound traffic: a preexisting connection
    # to a forbidden address must not survive policy installation.
    rules.extend(["-A OUTPUT -j REJECT --reject-with icmp-port-unreachable", "COMMIT", ""])
    return "\n".join(rules)


def render_ipv6_policy() -> str:
    """Block all external IPv6, including DNS and preexisting connections."""
    return "\n".join([
        "*filter",
        ":INPUT DROP [0:0]",
        ":FORWARD DROP [0:0]",
        ":OUTPUT DROP [0:0]",
        "-A INPUT -i lo -j ACCEPT",
        "-A OUTPUT -o lo -j ACCEPT",
        "-A OUTPUT -j REJECT --reject-with icmp6-port-unreachable",
        "COMMIT",
        "",
    ])


def resolve_hosts(hostnames: list[str]) -> list[str]:
    """Resolve A records once; failure for any hostname prevents startup."""
    addresses = set()
    for hostname in hostnames:
        try:
            answers = socket.getaddrinfo(
                hostname + ".", None,
                family=socket.AF_INET,
                type=socket.SOCK_STREAM,
                proto=socket.IPPROTO_TCP,
            )
            resolved = {str(ipaddress.IPv4Address(answer[4][0])) for answer in answers}
        except (socket.gaierror, ValueError) as error:
            raise RuntimeError(f"Could not resolve IPv4 addresses for {hostname}") from error
        if not resolved:
            raise RuntimeError(f"No IPv4 addresses found for {hostname}")
        addresses.update(resolved)
    return sorted(addresses)


def main() -> None:
    # Close both families before reading configuration or attempting DNS. Each
    # restore transaction replaces only filter rules, never Docker's NAT rules.
    subprocess.run(
        [IPTABLES_RESTORE, "--wait"],
        input=render_ipv4_policy([], []), text=True, check=True,
    )
    subprocess.run(
        [IP6TABLES_RESTORE, "--wait"],
        input=render_ipv6_policy(), text=True, check=True,
    )
    hostnames = parse_allowlist(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    nameservers = parse_nameservers(RESOLV_CONF_PATH.read_text(encoding="utf-8"))
    subprocess.run(
        [IPTABLES_RESTORE, "--wait"],
        input=render_ipv4_policy(nameservers, []), text=True, check=True,
    )
    addresses = resolve_hosts(hostnames)
    subprocess.run(
        [IPTABLES_RESTORE, "--wait"],
        input=render_ipv4_policy(nameservers, addresses), text=True, check=True,
    )


if __name__ == "__main__":
    main()
