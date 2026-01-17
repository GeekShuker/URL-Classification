from redis import Redis
from rq import Queue
from .config import settings

def get_queue() -> Queue:
    redis_conn = Redis.from_url(settings.redis_url)
    return Queue(connection=redis_conn, default_timeout=60*60*6)
