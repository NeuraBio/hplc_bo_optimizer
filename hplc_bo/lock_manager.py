import getpass
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout


class LockAcquisitionError(Exception):
    pass


class LockManager:
    def __init__(
        self, study_name: str, lock_dir: str = "optuna_storage/locks", timeout_sec: float = 0.0
    ):
        self.study_name = study_name
        self.lock_dir = Path(lock_dir)
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.lock_dir / f"{study_name}.lock"
        self.meta_path = self.lock_dir / f"{study_name}.lockmeta.json"
        self.timeout_sec = timeout_sec
        self._lock = FileLock(self.lock_path)

    def _write_metadata(self):
        metadata = {
            "pid": os.getpid(),
            "user": getpass.getuser() if hasattr(getpass, "getuser") else "unknown",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "study": self.study_name,
        }
        with open(self.meta_path, "w") as f:
            json.dump(metadata, f)

    def _delete_metadata(self):
        if self.meta_path.exists():
            self.meta_path.unlink()

    def read_metadata(self):
        if self.meta_path.exists():
            with open(self.meta_path) as f:
                return json.load(f)
        return None

    @contextmanager
    def acquire(self):
        try:
            self._lock.acquire(timeout=self.timeout_sec)
            self._write_metadata()
            yield
        except Timeout as err:
            meta = self.read_metadata()
            who = f"{meta['user']} (PID {meta['pid']})" if meta else "another process"
            raise LockAcquisitionError(f"Study '{self.study_name}' is locked by {who}.") from err
        finally:
            if self._lock.is_locked:
                self._lock.release()
                self._delete_metadata()
