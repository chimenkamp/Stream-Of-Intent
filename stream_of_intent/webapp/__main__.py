from __future__ import annotations

import argparse

from stream_of_intent.webapp.db import init_db
from stream_of_intent.webapp.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream of Intent — Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8050, help="Port (default: 8050)")
    parser.add_argument("--db-path", default=None, help="SQLite database path (default: webapp/models.db)")
    parser.add_argument("--debug", action="store_true", default=True, help="Run in debug mode")
    args = parser.parse_args()

    init_db(args.db_path)
    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
