import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from models import Base

# Use PostgreSQL in production (Render), fallback to SQLite for local Replit testing
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./saas_backend.db")

# Ensure it uses the asyncpg driver for PostgreSQL (handles both postgres:// and postgresql://)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

# Fix for asyncpg: it expects 'ssl=require' instead of 'sslmode=require'
if "sslmode=require" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("sslmode=require", "ssl=require")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
