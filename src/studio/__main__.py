import logging
import os
import sys
import threading
import webbrowser

from . import model_settings
from .app import create_app
from .config import LOOPBACK, Config

NO_AUTH_FLAG = "I_UNDERSTAND_THERE_IS_NO_AUTH"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    model_settings.install_log_redaction()
    cfg = Config.from_env()

    # There is no authentication anywhere in this app. On loopback that is
    # fine; on 0.0.0.0 anyone on the network can read your transcripts and
    # spend your API key. So a non-loopback bind is refused unless asked for
    # in so many words.
    if cfg.host not in LOOPBACK and os.environ.get(NO_AUTH_FLAG) != "1":
        print(
            f"\n  Refusing to listen on {cfg.host}: this app has no login, so "
            "anyone who can reach\n  the port could read your data and spend "
            "your API key.\n\n"
            f"  To do it anyway, set {NO_AUTH_FLAG}=1.\n",
            file=sys.stderr,
        )
        sys.exit(2)

    app = create_app(cfg)
    url = f"http://{'127.0.0.1' if cfg.host in ('0.0.0.0', '::') else cfg.host}:{cfg.port}"
    print(f"\n  EduBehaviors Studio → {url}")
    print(f"  Data directory:       {cfg.home}\n")
    if not model_settings.ready():
        print("  No verified models yet. Open the Models page, paste an API key "
              "and press Test connection.\n")
    if os.environ.get("NO_BROWSER") != "1":
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=cfg.host, port=cfg.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
