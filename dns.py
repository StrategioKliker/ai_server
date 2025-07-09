import os 
import time 
import socket 

_cache = {}

def resolve(host: str) -> str:
    env_var = os.getenv(f"{host.upper().replace('-', '_')}_IP")
    if env_var: 
        return env_var
    
    if host in _cache: 
        return _cache[host]
    
    last_err = None 
    for _ in range(3):
        try: 
            ip = socket.gethostbyname(host)
            _cache[host] = ip 
            return ip 
        except socket.gaierror as e: 
            last_err = e
            time.sleep(1)

    raise last_err