import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

# Absolute database path relative to this backend directory ensures consistent access
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "wardrobe.db")
DATABASE_URL = f"sqlite:///{DB_FILE}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db():
    """Create all tables and automatically migrate any missing columns in SQLite."""
    Base.metadata.create_all(bind=engine)

    # Inspect each table defined in SQLAlchemy models and safely add missing columns
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL;"))
        conn.execute(text("PRAGMA busy_timeout=5000;"))
        conn.commit()
        for table_name, table in Base.metadata.tables.items():
            result = conn.execute(text(f"PRAGMA table_info({table_name})"))
            existing_columns = {row[1] for row in result.fetchall()}

            for col in table.columns:
                if col.name not in existing_columns:
                    type_str = str(col.type)
                    default_clause = ""
                    if col.default is not None and hasattr(col.default, "arg"):
                        if isinstance(col.default.arg, str):
                            default_clause = f" DEFAULT '{col.default.arg}'"
                        elif isinstance(col.default.arg, (int, float)):
                            default_clause = f" DEFAULT {col.default.arg}"
                        elif isinstance(col.default.arg, bool):
                            default_clause = f" DEFAULT {1 if col.default.arg else 0}"

                    stmt = f"ALTER TABLE {table_name} ADD COLUMN {col.name} {type_str}{default_clause}"
                    conn.execute(text(stmt))
                    conn.commit()

