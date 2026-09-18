# Database schema

The application currently creates its PostgreSQL tables with SQLAlchemy `Base.metadata.create_all()` during startup.

For a production deployment with independent schema versioning, add Alembic migrations before making schema changes. The canonical SQLAlchemy models are in `app/database.py`.
