"""Relational metadata store (SQLite by default; PostgreSQL via DATABASE_URL).

Large intermediate data (records, candidate pairs, features) stays in columnar/pickled run artifacts;
the database holds queryable metadata: datasets, runs, metrics, accepted matches, model versions.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, String, Text,
                        create_engine)
from sqlalchemy.orm import declarative_base, sessionmaker

from ..core.settings import settings

engine = create_engine(settings.database_url, future=True,
                       connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
Base = declarative_base()


def now() -> dt.datetime:
    return dt.datetime.now().replace(microsecond=0)


class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(String(64), primary_key=True)
    name = Column(String(200), nullable=False)
    kind = Column(String(20), nullable=False)          # competition | synthetic | upload
    root = Column(Text, nullable=False)                # directory containing <split>/ folders
    splits = Column(JSON, default=list)
    created = Column(DateTime, default=now)
    description = Column(Text, default="")
    profile = Column(JSON, nullable=True)


class ResolutionRun(Base):
    __tablename__ = "resolution_runs"
    id = Column(String(32), primary_key=True)          # RUN-YYYY-NNNNN
    dataset_id = Column(String(64), ForeignKey("datasets.id"), nullable=False)
    split = Column(String(20), nullable=False)
    model_version = Column(String(64))
    status = Column(String(20), default="QUEUED")      # QUEUED | RUNNING | COMPLETED | FAILED
    stage = Column(String(120), default="")
    progress = Column(Integer, default=0)
    created = Column(DateTime, default=now)
    finished = Column(DateTime, nullable=True)
    runtime_s = Column(Float, nullable=True)
    error = Column(Text, nullable=True)
    summary = Column(JSON, nullable=True)


class Metric(Base):
    __tablename__ = "metrics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(32), ForeignKey("resolution_runs.id"), index=True)
    name = Column(String(80))
    value = Column(Float)


class Match(Base):
    __tablename__ = "matches"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(32), ForeignKey("resolution_runs.id"))
    s1_id = Column(String(64))
    other_id = Column(String(64))
    target = Column(String(4))
    confidence = Column(Float)
    correct = Column(Boolean, nullable=True)
    __table_args__ = (Index("ix_match_run_s1", "run_id", "s1_id"), Index("ix_match_run_other", "run_id", "other_id"))


class TrainingJob(Base):
    __tablename__ = "training_jobs"
    id = Column(String(40), primary_key=True)
    model = Column(String(40))
    status = Column(String(20), default="QUEUED")
    stage = Column(String(120), default="")
    created = Column(DateTime, default=now)
    finished = Column(DateTime, nullable=True)
    version = Column(String(64), nullable=True)
    error = Column(Text, nullable=True)


def init_db() -> None:
    Base.metadata.create_all(engine)
