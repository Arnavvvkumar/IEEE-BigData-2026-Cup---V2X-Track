import sys
import time

import carla

port = int(sys.argv[1]) if len(sys.argv) > 1 else 2200
deadline = time.time() + float(sys.argv[2] if len(sys.argv) > 2 else 600)
while time.time() < deadline:
    try:
        c = carla.Client('127.0.0.1', port)
        c.set_timeout(15)
        print('server', c.get_server_version(), 'map', c.get_world().get_map().name, flush=True)
        sys.exit(0)
    except RuntimeError as e:
        print('waiting:', str(e)[:80], flush=True)
        time.sleep(15)
sys.exit(1)
