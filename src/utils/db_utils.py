import os
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


class DatabaseConnection:
    def __init__(self, conn_str: str | None = None):
        if isinstance(conn_str, type(None)):
            self.conn_str = os.getenv("DATABASE_URL")
        else:
            self.conn_str = conn_str

        self.conn: Engine | None = None

    def __enter__(self) -> Engine:
        self.engine = create_engine(self.conn_str)
        return self.engine

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.engine:
            self.engine.dispose()
