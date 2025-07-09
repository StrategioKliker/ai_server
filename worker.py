import os
from time import sleep
from rq import Worker, Queue
from redis import Redis
from redis.exceptions import ConnectionError

# To give some breathing room to the internal DNS cache and model-server 
class LazyWorker(Worker):
    def execute_job(self, job, queue):
        super().execute_job(job, queue)
        if job.is_finished:
            print("[WORKER] Job completed, taking a nap...", flush=True)
            sleep(10)


listen = ['default']
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')

if __name__ == '__main__':
    while True: 
        try: 
            conn = Redis.from_url(redis_url, socket_timeout=None, retry_on_timeout=True)
            queue = Queue('default', connection=conn)
            worker = LazyWorker([queue], connection=conn)
            worker.work()
        except ConnectionError as e: 
            print(f"[WORKER] Redis unavailable, retrying in 2s... ({e})", flush=True)
            sleep(2)


