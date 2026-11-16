"""Run a single-node command under the shared cache's exclusive session lease."""
import argparse
import fcntl
import os
from pathlib import Path
import socket
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A command is required")
    if not os.environ.get("HIP_ORB_CACHE_LOCAL_LOCK_ROOT"):
        parser.error("A session lease requires HIP_ORB_CACHE_LOCAL_LOCK_ROOT; shared mode locks per transaction")
    root = Path(os.environ["HIP_ORB_CACHE_ROOT"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".cache-lock").open("a+") as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = dict(os.environ, HIP_ORB_CACHE_GLOBAL_LEASE="1",
                   HIP_ORB_CACHE_LEASE_HOST=socket.gethostname())
        # Inherited descriptor keeps the lease alive if this wrapper exits first.
        result = subprocess.run(command, env=env, pass_fds=(lease.fileno(),))
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
