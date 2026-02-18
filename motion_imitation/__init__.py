"""Top-level package setup for motion_imitation."""

import logging
import os
import warnings


def _configure_tf_warning_policy():
  """Silence TensorFlow/stable-baselines warning noise by default.

  Set `MOTION_IMITATION_SHOW_TF_WARNINGS=1` to opt out.
  """
  if os.environ.get("MOTION_IMITATION_SHOW_TF_WARNINGS", "0") in ("1", "true", "TRUE", "yes", "YES"):
    return

  os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
  warnings.filterwarnings("ignore", module=r"tensorflow(\..*)?$")
  warnings.filterwarnings("ignore", module=r"tensorboard(\..*)?$")
  warnings.filterwarnings("ignore", module=r"stable_baselines(\..*)?$")
  warnings.filterwarnings("ignore", message=r".*The TensorFlow contrib module.*")

  # Works whether TensorFlow is imported now or later.
  logging.getLogger("tensorflow").setLevel(logging.ERROR)
  logging.getLogger("absl").setLevel(logging.ERROR)


_configure_tf_warning_policy()
