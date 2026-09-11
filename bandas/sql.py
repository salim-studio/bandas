"""bandas.sql — query + persist DataFrames with real databases.

- DataFrame.sql("SELECT ...") : in-memory SQL via sqlite (zero extra deps)
- read_sql / to_sql : sqlite3 builtin, sqlalchemy/duckdb/postgres when installed
- Database class : unified connector (sqlite, duckdb, postgres, mysql, sqlalchemy URL)
"""
from __future__ import annotations
import sqlite3
import numpy as np


def _frames_to_sqlite(frames: dict, conn):
    import pandas as pd
    for name, f in frames.items():
        pdf = f.to_pandas() if hasattr(f, "to_pandas") else pd.DataFrame(f)
        pdf.to_sql(name, conn, if_exists="replace", index=False)


def sql_query(query: str, tables: dict):
    """Run SQL over bandas DataFrames given as {table_name: DataFrame}."""
    from .dataframe import DataFrame
    conn = sqlite3.connect(":memory:")
    try:
        _frames_to_sqlite(tables, conn)
        import pandas as pd
        pdf = pd.read_sql_query(query, conn)
        return DataFrame(pdf)
    finally:
        conn.close()


def read_sql(query, con, **kw):
    """read_sql(query, con) — con: sqlite3 conn, sqlalchemy engine/URL str, or DB-API conn."""
    from .dataframe import DataFrame
    import pandas as pd
    if isinstance(con, str):
        # URL or sqlite path
        if con.startswith("sqlite:///") or con.endswith((".db", ".sqlite", ".sqlite3")) or "://" not in con:
            path = con.replace("sqlite:///", "")
            conn = sqlite3.connect(path)
            try:
                return DataFrame(pd.read_sql_query(query, conn, **kw))
            finally:
                conn.close()
        try:
            from sqlalchemy import create_engine
            eng = create_engine(con)
            with eng.connect() as c:
                return DataFrame(pd.read_sql_query(query, c, **kw))
        except ImportError as e:
            raise ImportError(f"SQL URL needs sqlalchemy: {e}")
    return DataFrame(pd.read_sql_query(query, con, **kw))


def to_sql(df, name, con, if_exists="replace", index=False, **kw):
    pdf = df.to_pandas()
    if isinstance(con, str):
        path = con.replace("sqlite:///", "")
        conn = sqlite3.connect(path)
        try:
            return pdf.to_sql(name, conn, if_exists=if_exists, index=index, **kw)
        finally:
            conn.close()
    return pdf.to_sql(name, con, if_exists=if_exists, index=index, **kw)


class Database:
    """Unified DB connector: sqlite / duckdb / postgres / mysql / sqlalchemy.

    db = Database("sqlite:///app.db")  # or Database("sqlite:///:memory:")
    db = Database.sqlite("/tmp/a.db")  # helpers below
    df = db.query("SELECT * FROM t")
    db.write(df, "t")
    db.tables()  /  db.execute("CREATE TABLE ...")
    """

    def __init__(self, url_or_conn, **engine_kw):
        self.url = url_or_conn if isinstance(url_or_conn, str) else None
        self.conn = None if isinstance(url_or_conn, str) else url_or_conn
        self._engine = None
        self._engine_kw = engine_kw

    @classmethod
    def sqlite(cls, path=":memory:"): return cls(f"sqlite:///{path}" if path != ":memory:" else "sqlite:///:memory:")
    @classmethod
    def duckdb(cls, path=":memory:"):
        try:
            import duckdb
            return cls(duckdb.connect(path))
        except ImportError as e:
            raise ImportError(f"duckdb not installed: {e}")
    @classmethod
    def postgres(cls, **kw):
        try:
            from sqlalchemy import create_engine
        except ImportError as e:
            raise ImportError(f"postgres needs sqlalchemy+psycopg: {e}")
        url = f"postgresql://{kw.get('user','postgres')}:{kw.get('password','')}@{kw.get('host','localhost')}:{kw.get('port',5432)}/{kw.get('db','postgres')}"
        return cls(url)
    @classmethod
    def mysql(cls, **kw):
        try:
            from sqlalchemy import create_engine  # noqa
        except ImportError as e:
            raise ImportError(f"mysql needs sqlalchemy+pymysql: {e}")
        url = f"mysql+pymysql://{kw.get('user','root')}:{kw.get('password','')}@{kw.get('host','localhost')}:{kw.get('port',3306)}/{kw.get('db','test')}"
        return cls(url)

    def _connect(self):
        if self.conn is not None: return self.conn
        assert self.url
        if self.url == "sqlite:///:memory:" or self.url == ":memory:":
            self.conn = sqlite3.connect(":memory:"); return self.conn
        if self.url.startswith("sqlite") or self.url.endswith((".db", ".sqlite", ".sqlite3")) or "://" not in self.url:
            self.conn = sqlite3.connect(self.url.replace("sqlite:///", "")); return self.conn
        try:
            from sqlalchemy import create_engine
            if self._engine is None: self._engine = create_engine(self.url, **self._engine_kw)
            return self._engine
        except ImportError as e:
            raise ImportError(f"need sqlalchemy for {self.url}: {e}")

    def query(self, sql, **kw):
        from .dataframe import DataFrame
        import pandas as pd
        c = self._connect()
        if isinstance(c, sqlite3.Connection):
            return DataFrame(pd.read_sql_query(sql, c, **kw))
        try:
            with c.connect() as conn:
                return DataFrame(pd.read_sql_query(sql, conn, **kw))
        except AttributeError:
            return DataFrame(pd.read_sql_query(sql, c, **kw))

    def execute(self, sql, params=None):
        c = self._connect()
        if isinstance(c, sqlite3.Connection):
            cur = c.execute(sql, params or []); c.commit(); return cur
        try:
            from sqlalchemy import text
            with c.begin() as conn: return conn.execute(text(sql), params or {})
        except AttributeError:
            cur = c.cursor(); cur.execute(sql, params or []); c.commit(); return cur

    def write(self, df, table, if_exists="replace", index=False):
        c = self._connect()
        return to_sql(df, table, c, if_exists=if_exists, index=index)

    def tables(self):
        try:
            df = self.query("SELECT name FROM sqlite_master WHERE type='table'")
            return list(np.asarray(df._cols[df._columns[0]]))
        except Exception:
            try:
                df = self.query("SHOW TABLES")
                return list(np.asarray(df._cols[df._columns[0]]))
            except Exception:
                return []

    def close(self):
        try:
            if self.conn: self.conn.close()
        except Exception: pass
        try:
            if self._engine: self._engine.dispose()
        except Exception: pass
