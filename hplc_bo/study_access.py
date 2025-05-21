import logging

from optuna.study import Study

from hplc_bo.gradient_utils import TrialRecord
from hplc_bo.lock_manager import LockAcquisitionError, LockManager
from hplc_bo.optimizer import suggest_params

logger = logging.getLogger(__name__)


class StudyAccess:
    def __init__(self, study: Study, lock: LockManager, study_name: str):
        self.study = study
        self.lock = lock
        self.study_name = study_name

    def ask(self) -> TrialRecord:
        try:
            with self.lock.acquire():
                trial = self.study.ask()
                params = suggest_params(trial)
                record = TrialRecord.from_params(trial.number, params)
                record.save(self.study_name)
                return record
        except LockAcquisitionError as e:
            logger.error(f"[Lock Error - ask()] {e}")
            raise

    def tell(self, record: TrialRecord, extra_attrs: dict = None):
        try:
            with self.lock.acquire():
                if extra_attrs:
                    # Optional logging or usage only, can't actually set them with trial_id API
                    for k, v in extra_attrs.items():
                        print(f"[info] Cannot set attr '{k, v}' when using trial_id-based tell()")

                self.study.tell(record.trial_number, record.score)
                record.update_result(self.study_name)

        except LockAcquisitionError as e:
            logger.error(f"[Lock Error - tell()] Trial #{record.trial_number} — {e}")
            raise
