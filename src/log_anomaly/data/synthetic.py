"""Generate a labelled synthetic host-log corpus.

The generator writes two files that always travel together:

* a log file in the same BSD syslog shape as the Loghub Linux corpus, so the
  parser path is identical for synthetic and real data;
* a labels CSV with one row per log line, giving the ground truth.

**This corpus is synthetic.** The background traffic is modelled on the shape
of the real Loghub corpora (working-hours SSH logins, mistyped passwords, a
low-rate background internet scan) but every byte is produced by this
generator. The labels describe the generator's intent, not observed reality.

Label vocabulary
----------------
``benign``
    Ordinary activity, including failed logins caused by users mistyping
    their password.
``background_scan``
    Low-rate password probing from a fixed set of source addresses. This is
    real malicious activity that exists in the background of any
    internet-facing host; it is labelled as an attack, not as noise.
``brute_force``, ``credential_stuffing``, ``new_host_login``,
``off_hours_login``, ``privilege_escalation``, ``rare_process``,
``data_exfiltration``, ``lateral_movement``
    Deliberately injected attack families.

Label scope
-----------
An attack label covers the events that *evidence* the intrusion: the
authentication or privilege-escalation event, and the session-open record
that belongs to the same login. The process tree, network connections and
session close that follow are ordinary activity performed in an unusual
context and stay labelled ``benign``. Labelling them as attacks would inflate
recall with events that are, on their own, indistinguishable from normal.
"""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

#: Attack families the generator can inject, in the order they are reported.
ATTACK_FAMILIES: tuple[str, ...] = (
    "background_scan",
    "brute_force",
    "credential_stuffing",
    "new_host_login",
    "off_hours_login",
    "privilege_escalation",
    "rare_process",
    "data_exfiltration",
    "lateral_movement",
)

_FIRST_NAMES = [
    "alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi", "ivan",
    "judy", "mallory", "niaj", "olivia", "peggy", "rupert", "sybil", "trent",
    "victor", "walter", "wendy", "xavier", "yves", "zoe", "amir", "bianca",
    "carlos", "dmitri", "elena", "farid", "gita", "hugo", "ines", "jasper",
    "kiran", "lena", "mateo", "nadia", "omar", "petra", "quinn", "ravi",
    "sofia", "tomas", "ulf", "vera", "wes", "xiu", "yara", "zane",
]
_LAST_NAMES = [
    "adams", "brown", "clarke", "diaz", "evans", "fisher", "gomez", "hansen",
    "ibarra", "jones", "kowalski", "lindqvist", "moreau", "nakamura", "okafor",
    "pereira", "quinn", "rossi", "silva", "tanaka", "ueda", "varga", "walsh",
    "xiong", "yilmaz", "zhang",
]


def _build_user_names(count: int) -> list[str]:
    """A deterministic pool of ``count`` distinct account names."""
    names: list[str] = []
    for first in _FIRST_NAMES:
        for last in _LAST_NAMES:
            names.append(f"{first}.{last}")
            if len(names) == count:
                return names
    return names[:count]


_COMMON_USERNAMES = [
    "root", "admin", "administrator", "test", "guest", "oracle", "postgres",
    "mysql", "user", "ubuntu", "pi", "ftp", "www", "git", "deploy",
]

#: Normal process tree: parent executable -> [(child executable, weight)].
_NORMAL_CHILDREN: dict[str, list[tuple[str, float]]] = {
    "sshd": [("/bin/bash", 1.0)],
    "/bin/bash": [
        ("/usr/bin/python3", 0.8),
        ("/usr/bin/git", 0.5),
        ("/usr/bin/vim", 0.4),
        ("/usr/bin/less", 0.3),
    ],
    "nginx": [("nginx", 1.0), ("/usr/bin/curl", 0.2)],
    "cron": [("/usr/sbin/logrotate", 0.5), ("/usr/bin/find", 0.4)],
    "systemd": [("/usr/sbin/sshd", 1.0), ("/usr/bin/containerd", 0.3)],
    "containerd": [("/usr/bin/docker", 0.3)],
}

#: Every (parent, child) pair that legitimate traffic produces.
_NORMAL_PAIRS: frozenset[tuple[str, str]] = frozenset(
    (parent, child) for parent, children in _NORMAL_CHILDREN.items() for child, _ in children
)

#: Child programs that legitimate traffic never executes.
_NOVEL_CHILDREN: tuple[str, ...] = ("/bin/nc", "/usr/bin/perl", "/usr/bin/wget", "/bin/sh")

#: Every child program that legitimate traffic executes somewhere.
_ALL_NORMAL_CHILDREN: tuple[str, ...] = tuple(
    sorted({child for children in _NORMAL_CHILDREN.values() for child, _ in children})
)

_NORMAL_DESTINATIONS: tuple[tuple[str, int], ...] = (
    ("10.20.0.10", 443),
    ("10.20.0.11", 443),
    ("10.20.0.12", 5432),
    ("151.101.1.140", 443),
    ("104.18.32.7", 443),
    ("8.8.8.8", 53),
)


@dataclass(frozen=True)
class SyntheticConfig:
    """Knobs for the synthetic corpus. Defaults are what the report uses."""

    days: int = 60
    hosts: int = 6
    users: int = 120
    seed: int = 20240614
    start: str = "2024-03-04"
    #: Password probing from fixed source addresses, failures per hour per address.
    background_scan_rate: float = 3.0
    background_scan_ips: int = 3
    #: Share of users whose logins fall outside the 09:00-18:00 office window.
    night_shift_share: float = 0.2
    brute_force_campaigns: int = 8
    brute_force_attempts: int = 60
    credential_stuffing_campaigns: int = 6
    credential_stuffing_accounts: int = 25
    new_host_logins: int = 25
    off_hours_logins: int = 25
    privilege_escalation_campaigns: int = 8
    rare_process_events: int = 20
    exfiltration_events: int = 10
    lateral_movement_campaigns: int = 6

    @property
    def start_datetime(self) -> datetime:
        return datetime.fromisoformat(self.start).replace(tzinfo=UTC)


@dataclass
class _Event:
    ts: datetime
    host: str
    program: str
    pid: int
    event_type: str
    message: str
    label: int = 0
    attack_type: str = "benign"
    user: str = ""
    src_ip: str = ""
    dst_ip: str = ""
    dst_port: int = 0
    process: str = ""
    parent_process: str = ""
    bytes_sent: int = 0
    bytes_received: int = 0
    description: str = ""
    extra: dict[str, object] = field(default_factory=dict)


class _Generator:
    """Builds the event list for one synthetic corpus."""

    def __init__(self, config: SyntheticConfig) -> None:
        self.config = config
        self.random = random.Random(config.seed)
        self.events: list[_Event] = []
        self.pid = 1000
        self.host_names = [f"host{n:02d}" for n in range(1, config.hosts + 1)]
        self.host_ips = {
            host: f"10.20.{index}.{10 + (index % 200)}"
            for index, host in enumerate(self.host_names, start=1)
        }
        self.users = _build_user_names(config.users)
        self.user_home = {user: self.random.choice(self.host_names) for user in self.users}
        self.user_hosts: dict[str, list[str]] = {}
        for user in self.users:
            home = self.user_home[user]
            extra = [h for h in self.host_names if h != home and self.random.random() < 0.25]
            self.user_hosts[user] = [home, *extra]
        self.night_shift = {
            user: self.random.random() < config.night_shift_share for user in self.users
        }
        self.scanner_ips = [
            f"45.146.{index}.{self.random.randint(2, 250)}"
            for index in range(config.background_scan_ips)
        ]
        #: (host, executable) -> pids seen for that executable, newest last.
        self.program_pids: dict[tuple[str, str], list[int]] = {}

    # -- helpers ---------------------------------------------------------
    def next_pid(self) -> int:
        self.pid += 1
        return self.pid

    def random_second_in(self, day: datetime, hour: int) -> datetime:
        return day + timedelta(
            hours=hour, minutes=self.random.randint(0, 59), seconds=self.random.randint(0, 59)
        )

    def user_source_ip(self, user: str) -> str:
        return self.host_ips[self.user_home[user]]

    def _weighted_choice(self, options: list[tuple[str, float]]) -> str:
        total = sum(weight for _, weight in options)
        pick = self.random.uniform(0, total)
        cumulative = 0.0
        for value, weight in options:
            cumulative += weight
            if pick <= cumulative:
                return value
        return options[-1][0]

    # -- background ------------------------------------------------------
    def add_login(
        self,
        ts: datetime,
        user: str,
        host: str,
        src_ip: str,
        label: int = 0,
        attack_type: str = "benign",
        description: str = "",
    ) -> None:
        """Emit a successful login.

        Only the authentication and session-open records carry the attack
        label; the session close and the process tree stay benign.
        """
        pid = self.next_pid()
        self.events.append(
            _Event(
                ts=ts,
                host=host,
                program="sshd",
                pid=pid,
                event_type="auth_success",
                message=(
                    f"Accepted password for {user} from {src_ip} "
                    f"port {self.random.randint(30000, 65000)} ssh2"
                ),
                label=label,
                attack_type=attack_type,
                user=user,
                src_ip=src_ip,
                description=description,
            )
        )
        self.events.append(
            _Event(
                ts=ts + timedelta(seconds=1),
                host=host,
                program="sshd",
                pid=pid,
                event_type="session_open",
                message=f"pam_unix(sshd:session): session opened for user {user} by (uid=0)",
                label=label,
                attack_type=attack_type,
                user=user,
                src_ip=src_ip,
                description=description,
            )
        )
        self._add_process_tree(ts + timedelta(seconds=2), user, host)
        self.events.append(
            _Event(
                ts=ts + timedelta(minutes=self.random.randint(5, 90)),
                host=host,
                program="sshd",
                pid=pid,
                event_type="session_close",
                message=f"pam_unix(sshd:session): session closed for user {user}",
                user=user,
                src_ip=src_ip,
            )
        )

    def _add_process_tree(self, ts: datetime, user: str, host: str) -> None:
        """Emit a legitimate process chain, keeping parent/child pairs valid."""
        parent = "sshd"
        executable = "/bin/bash"
        for _ in range(self.random.randint(1, 3)):
            pid = self.next_pid()
            ppid = self.program_pids.get((host, parent), [1])[-1]
            self.events.append(
                _Event(
                    ts=ts,
                    host=host,
                    program="process",
                    pid=pid,
                    event_type="process_exec",
                    message=f"user={user} pid={pid} ppid={ppid} exe={executable}",
                    user=user,
                    process=executable,
                    parent_process=parent,
                    extra={"ppid": ppid},
                )
            )
            self.program_pids.setdefault((host, executable), []).append(pid)
            ts = ts + timedelta(milliseconds=self.random.randint(50, 900))
            children = _NORMAL_CHILDREN.get(executable)
            if not children:
                break
            parent = executable
            executable = self._weighted_choice(children)

    def add_connection(
        self,
        ts: datetime,
        user: str,
        host: str,
        program: str,
        dst: str,
        port: int,
        sent: int,
        received: int,
        label: int = 0,
        attack_type: str = "benign",
        description: str = "",
    ) -> None:
        pid = self.next_pid()
        self.events.append(
            _Event(
                ts=ts,
                host=host,
                program="conn",
                pid=pid,
                event_type="network_conn",
                message=(
                    f"user={user} exe={program} dst={dst}:{port} "
                    f"bytes_sent={sent} bytes_received={received}"
                ),
                label=label,
                attack_type=attack_type,
                user=user,
                dst_ip=dst,
                dst_port=port,
                process=program,
                bytes_sent=sent,
                bytes_received=received,
                description=description,
            )
        )

    def add_sudo(
        self,
        ts: datetime,
        user: str,
        host: str,
        ok: bool,
        label: int = 0,
        attack_type: str = "benign",
        description: str = "",
    ) -> None:
        pid = self.next_pid()
        if ok:
            message = f"{user} : TTY=pts/0 ; PWD=/home/{user} ; USER=root ; COMMAND=/bin/bash"
            event_type = "sudo_success"
        else:
            message = (
                f"pam_unix(sudo:auth): authentication failure; logname={user} uid=1000 "
                f"euid=0 tty=pts/0 ruser={user}  user=root"
            )
            event_type = "sudo_failure"
        self.events.append(
            _Event(
                ts=ts,
                host=host,
                program="sudo",
                pid=pid,
                event_type=event_type,
                message=message,
                label=label,
                attack_type=attack_type,
                user=user,
                description=description,
            )
        )

    def add_failure(
        self,
        ts: datetime,
        user: str,
        host: str,
        src_ip: str,
        label: int = 0,
        attack_type: str = "benign",
        description: str = "",
    ) -> None:
        pid = self.next_pid()
        if self.random.random() < 0.5:
            message = (
                f"Failed password for invalid user {user} from {src_ip} "
                f"port {self.random.randint(30000, 65000)} ssh2"
            )
        else:
            message = (
                "authentication failure; logname= uid=0 euid=0 tty=NODEVssh "
                f"ruser= rhost={src_ip}  user={user}"
            )
        self.events.append(
            _Event(
                ts=ts,
                host=host,
                program="sshd(pam_unix)",
                pid=pid,
                event_type="auth_failure",
                message=message,
                label=label,
                attack_type=attack_type,
                user=user,
                src_ip=src_ip,
                description=description,
            )
        )

    # -- background traffic ----------------------------------------------
    def build_background(self) -> None:
        config = self.config
        start = config.start_datetime
        for day_index in range(config.days):
            day = start + timedelta(days=day_index)
            for user in self.users:
                is_night = self.night_shift[user]
                hours = list(range(22, 24)) if is_night else list(range(8, 19))
                for _ in range(self.random.randint(1, 5)):
                    hour = self.random.choice(hours)
                    ts = self.random_second_in(day, hour)
                    host = self.random.choice(self.user_hosts[user])
                    self.add_login(ts, user, host, self.user_source_ip(user))
                    if self.random.random() < 0.3:
                        destination, port = self.random.choice(_NORMAL_DESTINATIONS)
                        self.add_connection(
                            ts + timedelta(minutes=self.random.randint(1, 30)),
                            user,
                            host,
                            "/usr/bin/python3",
                            destination,
                            port,
                            self.random.randint(500, 200_000),
                            self.random.randint(500, 900_000),
                        )
                    if self.random.random() < 0.08:
                        self.add_sudo(
                            ts + timedelta(minutes=self.random.randint(1, 20)), user, host, ok=True
                        )
                # users who mistype their password: benign failures in short bursts
                if self.random.random() < 0.2:
                    attempts = self.random.randint(1, 3)
                    base = self.random_second_in(day, self.random.randint(8, 18))
                    host = self.user_home[user]
                    for attempt in range(attempts):
                        self.add_failure(
                            base + timedelta(seconds=attempt * 4),
                            user,
                            host,
                            self.user_source_ip(user),
                            description="mistyped password",
                        )
            # background internet scanning: real attacks, labelled as such
            for hour in range(24):
                for scanner in self.scanner_ips:
                    count = 0
                    while self.random.random() < config.background_scan_rate / 20:
                        count += 1
                    for _ in range(count):
                        ts = self.random_second_in(day, hour)
                        target = self.random.choice(_COMMON_USERNAMES + self.users[:5])
                        self.add_failure(
                            ts,
                            target,
                            self.random.choice(self.host_names),
                            scanner,
                            label=1,
                            attack_type="background_scan",
                            description="background internet password probing",
                        )

    # -- injected attacks -------------------------------------------------
    def inject(self) -> None:
        config = self.config
        start = config.start_datetime
        span = timedelta(days=config.days)

        def moment() -> datetime:
            return start + timedelta(seconds=self.random.uniform(0, span.total_seconds()))

        for _ in range(config.brute_force_campaigns):
            ts = moment()
            victim = self.random.choice(self.users)
            src = f"185.220.{self.random.randint(1, 250)}.{self.random.randint(2, 250)}"
            host = self.user_home[victim]
            for attempt in range(config.brute_force_attempts):
                self.add_failure(
                    ts + timedelta(seconds=attempt * self.random.uniform(1.5, 4.0)),
                    victim,
                    host,
                    src,
                    label=1,
                    attack_type="brute_force",
                    description="password brute force against one account",
                )

        for _ in range(config.credential_stuffing_campaigns):
            ts = moment()
            src = f"194.26.{self.random.randint(1, 250)}.{self.random.randint(2, 250)}"
            host = self.random.choice(self.host_names)
            accounts = self.random.sample(
                self.users, k=min(config.credential_stuffing_accounts, len(self.users))
            )
            for account in accounts:
                for _ in range(2):
                    self.add_failure(
                        ts + timedelta(seconds=self.random.uniform(0, 900)),
                        account,
                        host,
                        src,
                        label=1,
                        attack_type="credential_stuffing",
                        description="one source spraying many accounts",
                    )

        for _ in range(config.new_host_logins):
            ts = moment()
            user = self.random.choice(self.users)
            unusual = [h for h in self.host_names if h != self.user_home[user]]
            host = self.random.choice(unusual)
            self.add_login(
                ts,
                user,
                host,
                f"203.0.113.{self.random.randint(2, 250)}",
                label=1,
                attack_type="new_host_login",
                description="account used on a host it never normally uses",
            )

        for _ in range(config.off_hours_logins):
            candidates = [u for u in self.users if not self.night_shift[u]]
            if not candidates:
                continue
            user = self.random.choice(candidates)
            host = self.user_home[user]
            ts = moment().replace(hour=self.random.randint(0, 5), minute=self.random.randint(0, 59))
            self.add_login(
                ts,
                user,
                host,
                f"198.51.100.{self.random.randint(2, 250)}",
                label=1,
                attack_type="off_hours_login",
                description="login outside the account's normal working hours",
            )

        for _ in range(config.privilege_escalation_campaigns):
            ts = moment()
            user = self.random.choice(self.users)
            host = self.user_home[user]
            for attempt in range(self.random.randint(2, 5)):
                self.add_sudo(
                    ts + timedelta(seconds=attempt * self.random.uniform(2, 8)),
                    user,
                    host,
                    ok=False,
                    label=1,
                    attack_type="privilege_escalation",
                    description="repeated failed privilege escalation",
                )
            self.add_sudo(
                ts + timedelta(seconds=30),
                user,
                host,
                ok=True,
                label=1,
                attack_type="privilege_escalation",
                description="privilege escalation succeeded after failures",
            )

        for _ in range(config.rare_process_events):
            ts = moment()
            user = self.random.choice(self.users)
            host = self.user_home[user]
            known = [(h, exe) for (h, exe) in self.program_pids if h == host]
            if not known:
                continue
            parent_host, parent = self.random.choice(known)
            child = self._pick_unusual_child(parent)
            parent_pid = self.random.choice(self.program_pids[(parent_host, parent)])
            pid = self.next_pid()
            self.events.append(
                _Event(
                    ts=ts,
                    host=host,
                    program="process",
                    pid=pid,
                    event_type="process_exec",
                    message=f"user={user} pid={pid} ppid={parent_pid} exe={child}",
                    label=1,
                    attack_type="rare_process",
                    description="program executed under an unusual parent",
                    user=user,
                    process=child,
                    parent_process=parent,
                    extra={"ppid": parent_pid},
                )
            )

        for _ in range(config.exfiltration_events):
            ts = moment()
            user = self.random.choice(self.users)
            host = self.user_home[user]
            destination = f"203.0.113.{self.random.randint(2, 250)}"
            self.add_connection(
                ts,
                user,
                host,
                "/usr/bin/curl",
                destination,
                443,
                self.random.randint(20_000_000, 90_000_000),
                self.random.randint(100_000, 2_000_000),
                label=1,
                attack_type="data_exfiltration",
                description="large outbound transfer to an unseen destination",
            )

        for _ in range(config.lateral_movement_campaigns):
            ts = moment()
            user = self.random.choice(self.users)
            src = f"192.0.2.{self.random.randint(2, 250)}"
            for host in self.host_names:
                self.add_login(
                    ts + timedelta(seconds=self.random.uniform(0, 1200)),
                    user,
                    host,
                    src,
                    label=1,
                    attack_type="lateral_movement",
                    description="one account authenticating to many hosts in a short window",
                )

    def _pick_unusual_child(self, parent: str) -> str:
        """Pick a child that legitimate traffic never runs under ``parent``.

        Half the time the child program is one that never appears at all;
        the other half it is a legitimate program that this particular parent
        never spawns, so only the *pair* is novel.
        """
        if self.random.random() < 0.5:
            return self.random.choice(list(_NOVEL_CHILDREN))
        normal_children = {child for child, _ in _NORMAL_CHILDREN.get(parent, [])}
        candidates = [child for child in _ALL_NORMAL_CHILDREN if child not in normal_children]
        return self.random.choice(candidates or list(_NOVEL_CHILDREN))

    def build(self) -> list[_Event]:
        self.build_background()
        self.inject()
        self.events.sort(key=lambda event: (event.ts, event.host, event.pid))
        return self.events


def render_syslog(event: _Event) -> str:
    """Render an event as a BSD syslog line."""
    stamp = event.ts.strftime("%b %d %H:%M:%S")
    pid = f"[{event.pid}]" if event.pid else ""
    return f"{stamp} {event.host} {event.program}{pid}: {event.message}"


def write_corpus(
    config: SyntheticConfig,
    log_path: str | Path,
    labels_path: str | Path | None = None,
    meta_path: str | Path | None = None,
) -> dict[str, object]:
    """Generate the corpus and write the log, labels and metadata files."""
    generator = _Generator(config)
    events = generator.build()

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("# SYNTHETIC LOG - generated by log-anomaly. Not real host telemetry.\n")
        for event in events:
            handle.write(render_syslog(event) + "\n")

    counts: dict[str, int] = {}
    for event in events:
        counts[event.attack_type] = counts.get(event.attack_type, 0) + 1

    summary: dict[str, object] = {
        "synthetic": True,
        "config": asdict(config),
        "events": len(events),
        "anchor_year": config.start_datetime.year,
        "attack_counts": counts,
        "base_rate": round(
            sum(1 for event in events if event.label) / max(len(events), 1), 6
        ),
    }

    if labels_path is not None:
        labels_path = Path(labels_path)
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        with labels_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "event_index",
                    "ts",
                    "label",
                    "attack_type",
                    "user",
                    "src_ip",
                    "host",
                    "event_type",
                    "description",
                ]
            )
            for index, event in enumerate(events):
                writer.writerow(
                    [
                        index,
                        event.ts.isoformat(),
                        event.label,
                        event.attack_type,
                        event.user,
                        event.src_ip,
                        event.host,
                        event.event_type,
                        event.description,
                    ]
                )

    if meta_path is not None:
        meta_path = Path(meta_path)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return summary


def generate(
    config: SyntheticConfig, log_path: str | Path, labels_path: str | Path
) -> dict[str, object]:
    """Convenience wrapper used by the CLI."""
    log = Path(log_path)
    return write_corpus(config, log, labels_path, log.with_suffix(log.suffix + ".meta.json"))


def read_labels(labels_path: str | Path) -> list[dict[str, object]]:
    """Read a labels CSV written by :func:`write_corpus`."""
    with Path(labels_path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def iter_attack_families() -> Iterable[str]:
    return ATTACK_FAMILIES
