from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Client(Base):
    __tablename__ = "clients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    api_key = Column(String, unique=True, index=True)

    # NEW USAGE TRACKING COLUMNS
    message_count = Column(Integer, default=0)
    last_active = Column(String, nullable=True)


class Website(Base):
    __tablename__ = "websites"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"))
    url = Column(String, index=True)
    bot_status = Column(String, default="pending")
