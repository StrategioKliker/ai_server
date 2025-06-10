import os
from time import sleep
from rq import Worker, Queue
from redis import Redis
from redis.exceptions import ConnectionError

listen = ['default']
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')

if __name__ == '__main__':
    while True: 
        try: 
            conn = Redis.from_url(redis_url, socket_timeout=None, retry_on_timeout=True)
            queue = Queue('default', connection=conn)
            worker = Worker([queue], connection=conn)
            worker.work()
        except ConnectionError as e: 
            print(f"[WORKER] Redis unavailable, retrying in 5s... ({e})")
            sleep(5)


