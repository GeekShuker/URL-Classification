from redis import Redis
from rq import Worker, Queue, Connection
from system.api.config import settings

def main():
    redis_conn = Redis.from_url(settings.redis_url)
    with Connection(redis_conn):
        q = Queue()
        w = Worker([q])
        w.work(with_scheduler=False)

if __name__ == "__main__":
    main()
