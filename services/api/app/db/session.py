from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row
from app.core.config import DATABASE_URL
from app.core.config import DATA_DATABASE_URL, META_DATABASE_URL

@contextmanager
def get_meta_conn():
    with psycopg.connect(META_DATABASE_URL, row_factory=dict_row) as conn:
        yield conn

@contextmanager
def get_data_conn():
    with psycopg.connect(DATA_DATABASE_URL, row_factory=dict_row) as conn:
        yield conn


@contextmanager
def get_conn():
    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        yield conn
