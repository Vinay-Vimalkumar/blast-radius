import atexit
import os
from pathlib import Path
import sys

if os.getenv("TPC_ORIGIN_STAGE"):
    stage = Path(os.environ["TPC_ORIGIN_STAGE"])
    target = os.environ["TPC_ORIGIN_FILE"]
    mod = target.removesuffix(".py").replace("/", ".").removesuffix(".__init__")

    def origin_at_exit():
        import hashlib
        module = sys.modules.get(mod)
        ok = module is not None and Path(module.__file__) == stage / target
        if ok:
            ok = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == os.environ["TPC_ORIGIN_HASH"]
        if ok and origin_calls:
            (stage / f"worker-{os.getpid()}").write_text(str(origin_calls))

    origin_calls = 0

    def origin_start(code, offset):
        global origin_calls
        if code.co_name == "mp_execute" and code.co_filename.endswith("/tests/test_executor.py"):
            origin_calls += 1
            return
        return sys.monitoring.DISABLE

    sys.monitoring.use_tool_id(4, "worker origin")
    sys.monitoring.register_callback(4, sys.monitoring.events.PY_START, origin_start)
    sys.monitoring.set_events(4, sys.monitoring.events.PY_START)
    atexit.register(origin_at_exit)
