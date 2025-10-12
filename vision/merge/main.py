# merge/main.py
from __future__ import annotations

import sys
from utils.logger import Logger
from utils.error_tracker import ErrorTracker, error_scope
from .config import build_defaults
from .pipeline import merge_capture_to_ply, run_trusskit


def main() -> int:
    """Entry point: install hooks, run merge + TrussKit."""
    log = Logger.get_logger("app.merge")

    # Global hooks: exceptions, SIGINT/SIGTERM, ESC/CTRL+C stop.
    ErrorTracker.install_excepthook()
    ErrorTracker.install_signal_handlers()
    ErrorTracker.install_keyboard_listener(stop_key="esc")

    app = build_defaults()
    log.info(f"[INIT] root={app.capture_root} img={app.img_dir_name}")

    with error_scope():
        ply = merge_capture_to_ply(app)
        run_trusskit(ply, app)
        log.info("[DONE] merge+trusskit succeeded")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as e:
        Logger.get_logger("app.merge").info(f"[EXIT] code={e.code}")
        raise
