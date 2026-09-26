"""Database settings read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DbSettings:
    """SQL Server connection settings.

    Env vars: MSSQL_SERVER, MSSQL_DATABASE, MSSQL_DRIVER and MSSQL_AUTH. MSSQL_AUTH defaults to
    Windows authentication; set it to "UID=sa;PWD=..." to use SQL logins instead.
    """

    server: str = "localhost"
    database: str = "week1_ai"
    driver: str = "ODBC Driver 17 for SQL Server"
    auth: str = "Trusted_Connection=yes"

    @classmethod
    def from_env(cls) -> DbSettings:
        return cls(
            server=os.getenv("MSSQL_SERVER", cls.server),
            database=os.getenv("MSSQL_DATABASE", cls.database),
            driver=os.getenv("MSSQL_DRIVER", cls.driver),
            auth=os.getenv("MSSQL_AUTH", cls.auth),
        )

    def conn_str(self, database: str | None = None) -> str:
        return (
            f"DRIVER={{{self.driver}}};SERVER={self.server};"
            f"DATABASE={database or self.database};{self.auth};TrustServerCertificate=yes;"
        )
