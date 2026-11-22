"""Rebuild the frontend replay's maps so that the replay also covers plan construction.

usage (inside the image, on the pinned tree, PYTHONHASHSEED=0):
    python -B build_fe_maps.py <out-dir>      write fe_baseline, fe_lines and fe_consumer (.json.gz) to <out-dir>
    python -B build_fe_maps.py --validate     replay the pinned tree against the maps next to this file

The maps this starts from (the three files next to this script) were first recorded with the replay ending at
the optimized tree and its rendering; their consumer lines are the lines four runs of the real test executed
outside parse_one, transpile, find_tables, optimize and the pickling. fe_replay.worker_role can now also build the
plan and digest it. This script can be run again on its own output: it sets aside the plan digests and the
admitted lines an earlier run recorded. It

  1. re-records the replay's artifacts in a fresh process, checks that every artifact of the old baseline is
     reproduced, and adds plan<i> (the plan digest of every executed query) and plan_cpu;
  2. traces sqlglot.executor.execute() for every executed query, once as the first task of a fresh process and
     once after all other queries, and attributes each sqlglot line to the step that ran it: U unpickle,
     T find_tables, E ensure_tables, S ensure_schema, O optimize, R the sql(pretty=True) rendering,
     P Plan(expression), I PythonExecutor(...), B .execute(plan), X execute()'s own body (every line of it:
     the replay calls optimize, sql and Plan itself and never runs that body), K the pickle round trip of the
     result. It also traces what runs before the first task: in the pytest process up to
     the end of setUpClass and the pickling of the pool's initializer arguments, and in a pool worker while it
     unpickles them and runs the initializer;
  3. writes fe_lines with "plan": for each line the queries whose rendering or plan construction runs it (the
     replay builds the plan for those), and with these masks or-ed into the old worker masks, which are
     otherwise kept as they were;
  4. writes fe_consumer without the lines that are not start-up lines, that the traces saw in no step other
     than U, T, O, R and P, and that every query runs in R and P whether it is the first task of its process
     or not. A line seen anywhere else (E, S, I, B, X, K or start-up) stays a consumer line. So does a line
     that some query runs in R or P only as a first task: the replay handles its queries in one process, where
     only the first of them is in that position.
"""
import gzip
import inspect
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.realpath(os.environ.get("SQLGLOT_ROOT", "/opt/sqlglot"))
PREFIX = os.path.join(ROOT, "sqlglot") + os.sep
SCRIPT = os.path.join(HERE, "build_fe_maps.py")
REPLAYED = frozenset("UTORP")
MASKED = "RP"


def load(name):
    with gzip.open(os.path.join(HERE, name), "rt", encoding="utf-8") as stream:
        return json.load(stream)


def dump(directory, name, value):
    data = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    with open(os.path.join(directory, name), "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
        stream.write(data)


def fresh(*words):
    """Run this script in a new interpreter and return the JSON object it prints last."""
    env = {**os.environ, "PYTHONPATH": ROOT, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run([sys.executable, "-B", SCRIPT, *words], env=env, cwd=ROOT, capture_output=True, text=True)
    if done.returncode != 0 or not done.stdout.strip():
        raise SystemExit(f"{' '.join(words)[:60]} failed ({done.returncode}): {done.stderr[-600:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def forked(function):
    """Run function() in a forked child and return its picklable result."""
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read)
        try:
            try:
                payload = pickle.dumps((True, function()))
            except BaseException as exc:
                payload = pickle.dumps((False, f"{type(exc).__name__}: {exc}"[:400]))
            with os.fdopen(write, "wb") as stream:
                stream.write(payload)
        finally:
            os._exit(0)
    os.close(write)
    with os.fdopen(read, "rb") as stream:
        data = stream.read()
    os.waitpid(pid, 0)
    done, value = pickle.loads(data)
    if not done:
        raise SystemExit(f"child failed: {value}")
    return value





def record(mode):
    """(a fresh interpreter) Run both roles as the replay child does, with nothing to compare against."""
    sys.path.insert(0, HERE)
    import fe_replay

    class Detailed(fe_replay.Stages):
        def __init__(self):
            super().__init__()
            self.detail = {}

        def __exit__(self, *exc):
            key = f"{self.kind}:{self.index}"
            self.detail[key] = self.detail.get(key, 0.0) + time.process_time() - self.began
            return super().__exit__(*exc)

        def report(self):
            return {**super().report(), "detail": {key: round(value, 4) for key, value in self.detail.items()}}

    old = fe_replay._load("fe_baseline.json.gz")
    fe_replay.child_setup({}, cpu_limit=0)
    pairs, schema = fe_replay.prepare()
    everything = list(range(len(pairs)))
    executed = [index for index in everything if pairs[index][0].get("execute")]
    if mode == "all":
        work, order, duck, session = everything, everything, executed, old["session_modules"]
        planned = frozenset(executed)
    elif mode == "plain":
        work, order, duck, session, planned = everything, everything, executed, old["session_modules"], frozenset()
    else:
        index = int(mode)
        work, order, duck, session, planned = [index], [index], [], None, frozenset([index])
    verdict, parent, worker, reports = fe_replay.run_roles(pairs, schema, order, duck, expected=None, work=work,
                                                         session=session, planned=planned,
                                                         stages_factory=Detailed)
    sys.stdout.write(json.dumps({"verdict": verdict, "parent": parent, "worker": worker, "reports": reports,
                                 "python": sys.version, "tree": fe_replay.tree_hash(fe_replay.Path(ROOT)),
                                 "executed": executed, "queries": len(pairs)}) + "\n")





class Tracer:
    def __init__(self):
        self.monitoring = sys.monitoring
        self.tool = self.monitoring.PROFILER_ID
        self.monitoring.use_tool_id(self.tool, "fe maps")
        self.lines = None
        self.monitoring.register_callback(self.tool, self.monitoring.events.LINE, self._line)

    def _line(self, code, line):
        name = code.co_filename
        if name.startswith(PREFIX):
            self.lines.add((name[len(ROOT) + 1:].replace(os.sep, "/"), line))
        return self.monitoring.DISABLE

    def run(self, function):
        self.lines = set()
        self.monitoring.restart_events()
        self.monitoring.set_events(self.tool, self.monitoring.events.LINE)
        try:
            function()
        finally:
            self.monitoring.set_events(self.tool, 0)
        return sorted(self.lines)


def startup(role, path):
    """(a fresh interpreter) Trace sqlglot from the first import to the first task of one process of the test."""
    tracer = Tracer()
    if role == "parent":
        def body():
            from tests.helpers import TPCDS_SCHEMA
            from tests.test_executor import TestExecutor, initializer

            TestExecutor.setUpClass()
            with open(path, "wb") as stream:
                pickle.dump((initializer, (TPCDS_SCHEMA, TestExecutor.tpcds_tables)), stream)
    else:
        def body():
            with open(path, "rb") as stream:
                initializer, initargs = pickle.load(stream)
            initializer(*initargs)
    sys.stdout.write(json.dumps({"lines": tracer.run(body)}) + "\n")
    sys.stdout.flush()
    os._exit(0)


def trace_steps():
    """{path: {line: {step: [cold mask, warm mask]}}} from the real execute() flow: the queries that run the line
    in that step as the first task of a fresh process, and after every other query in the same process."""
    from tests.helpers import TPCDS_SCHEMA, load_sql_fixture_pairs
    from tests.test_executor import TestExecutor

    import sqlglot.executor as executor_module
    from sqlglot import find_tables, parse_one
    from sqlglot.executor.python import PythonExecutor

    TestExecutor.setUpClass()
    tables = TestExecutor.tpcds_tables
    pairs = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
    executed = [index for index, (meta, _, _) in enumerate(pairs) if meta.get("execute")]

    blobs = forked(lambda: {index: pickle.dumps(parse_one(pairs[index][1])) for index in executed})
    execute = executor_module.execute
    code = execute.__code__
    real = PythonExecutor.execute
    source, first = inspect.getsourcelines(execute)
    step_of_line = {}
    for offset, text in enumerate(source):
        stripped = text.strip()
        for prefix, step in (("tables_ = ensure_tables(", "E"), ("schema = ensure_schema(", "S"),
                             ("expression = optimize(", "O"), ("plan = Plan(", "P"),
                             ("result = PythonExecutor(", "I")):
            if stripped.startswith(prefix):
                step_of_line[first + offset] = step
        if "expression.sql(pretty=True)" in stripped:
            step_of_line[first + offset] = "R"
    if sorted(step_of_line.values()) != ["E", "I", "O", "P", "R", "S"]:
        raise SystemExit(f"execute() does not have the expected statements: {step_of_line}")

    def traced(blob):
        monitoring = sys.monitoring
        tool = monitoring.PROFILER_ID
        monitoring.use_tool_id(tool, "fe steps")
        seen, step = {}, ["U"]

        def on_step(name):
            if name != step[0]:
                step[0] = name
                monitoring.restart_events()

        def on_line(code_, line):
            if code_ is code:
                on_step(step_of_line.get(line, "X"))
                seen.setdefault("X", set()).add(("sqlglot/executor/__init__.py", line))
                return None
            name = code_.co_filename
            if name.startswith(PREFIX):
                seen.setdefault(step[0], set()).add((name[len(ROOT) + 1:].replace(os.sep, "/"), line))
            return monitoring.DISABLE

        def inner(self, plan, *args, **kwargs):
            if step[0] == "I":
                on_step("B")
            return real(self, plan, *args, **kwargs)

        monitoring.register_callback(tool, monitoring.events.LINE, on_line)
        PythonExecutor.execute = inner
        monitoring.restart_events()
        monitoring.set_events(tool, monitoring.events.LINE)
        try:
            expression = pickle.loads(blob)
            on_step("T")
            wanted = {found.name: tables[found.name] for found in find_tables(expression)}
            on_step("X")
            result = execute(expression, schema=TPCDS_SCHEMA, tables=wanted)
            on_step("K")
            pickle.loads(pickle.dumps(result))
        finally:
            monitoring.set_events(tool, 0)
            monitoring.free_tool_id(tool)
            PythonExecutor.execute = real
        return {name: sorted(lines) for name, lines in seen.items()}

    def plain(blob):
        expression = pickle.loads(blob)
        wanted = {found.name: tables[found.name] for found in find_tables(expression)}
        pickle.loads(pickle.dumps(execute(expression, schema=TPCDS_SCHEMA, tables=wanted)))

    def warm_all():
        for index in executed:
            plain(blobs[index])
        return {index: traced(blobs[index]) for index in executed}

    masks = {}
    warm = forked(warm_all)
    for index in executed:
        for slot, found in enumerate((forked(lambda: traced(blobs[index])), warm[index])):
            for step, lines in found.items():
                for path, line in map(tuple, lines):
                    pair = masks.setdefault(path, {}).setdefault(line, {}).setdefault(step, [0, 0])
                    pair[slot] |= 1 << index
    return masks




def build(out):
    old_baseline, old_lines, old_consumer = load("fe_baseline.json.gz"), load("fe_lines.json.gz"), load("fe_consumer.json.gz")
    plain, planned = fresh("--record", "plain"), fresh("--record", "all")
    for run in (plain, planned):
        if (run["verdict"] != "ok" or run["python"] != old_baseline["python"] or run["tree"] != old_baseline["tree"]
                or run["executed"] != old_baseline["executed"] or run["queries"] != old_baseline["queries"]):
            raise SystemExit(f"recording does not match the old baseline's environment: {run['verdict']}")
    executed = plain["executed"]
    if plain["parent"] != old_baseline["parent"] or plain["reports"]["session_artifacts"] != old_baseline["parent"]:
        raise SystemExit("the parent artifacts of the old baseline are not reproduced")
    recorded = {name: value for name, value in old_baseline["worker"].items() if not name.startswith("plan")}
    if plain["worker"] != recorded:
        differing = sorted(name for name in plain["worker"].keys() | recorded.keys()
                           if plain["worker"].get(name) != recorded.get(name))
        raise SystemExit(f"the worker artifacts of the old baseline are not reproduced: {differing[:8]}")
    extra = {name: value for name, value in planned["worker"].items() if name not in plain["worker"]}
    moved = sorted(name for name, value in plain["worker"].items() if planned["worker"].get(name) != value)
    if sorted(extra) != sorted(f"plan{index}" for index in executed) or moved:
        raise SystemExit(f"building the plans changes other artifacts: {moved[:8]} / {sorted(extra)[:4]}")

    for index in executed:
        alone = fresh("--record", str(index))
        if (alone["verdict"] != "ok" or alone["worker"].get(f"plan{index}") != extra[f"plan{index}"]
                or alone["worker"].get("state") != plain["worker"]["state"]):
            raise SystemExit(f"query {index} alone gives another plan digest or state")
    detail = planned["reports"]["worker"]["detail"]
    plan_cpu = {str(index): round(detail.get(f"plan:{index}", 0.0) + detail.get(f"plandigest:{index}", 0.0), 4)
                for index in executed}
    baseline = {**old_baseline, "worker": {**old_baseline["worker"], **extra}, "plan_cpu": plan_cpu}

    masks = trace_steps()
    with tempfile.TemporaryDirectory() as scratch:
        path = os.path.join(scratch, "initargs.pickle")
        before_first_task = {}
        for role in ("parent", "worker"):
            for name, line in fresh("--startup", role, path)["lines"]:
                before_first_task.setdefault(name, set()).add(line)

    worker = {path: dict(table) for path, table in old_lines["worker"].items()}
    plan = {}
    for path, table in masks.items():
        for line, steps in table.items():
            built = 0
            for step in MASKED:
                cold, warm = steps.get(step, (0, 0))
                built |= cold | warm
            if built:
                merged = int(worker.get(path, {}).get(str(line), "0"), 16) | built
                worker.setdefault(path, {})[str(line)] = format(merged, "x")
                plan.setdefault(path, {})[str(line)] = format(built, "x")
    lines = {**old_lines, "worker": {path: dict(sorted(table.items(), key=lambda item: int(item[0])))
                                     for path, table in sorted(worker.items())},
             "plan": {path: dict(sorted(table.items(), key=lambda item: int(item[0])))
                      for path, table in sorted(plan.items())}}

    kept, admitted, first_only = {}, {}, 0
    outside = {path: set(listed) for path, listed in old_consumer["test_execute_tpcds"].items()}
    for path, listed in old_consumer.get("admitted", {}).items():
        outside.setdefault(path, set()).update(listed)
    for path, listed in sorted(outside.items()):
        early = before_first_task.get(path, ())
        for line in sorted(listed):
            steps = masks.get(path, {}).get(line, {})
            if steps and set(steps) <= REPLAYED and line not in early:
                if any(steps[step][0] & ~steps[step][1] for step in MASKED if step in steps):
                    first_only += 1
                    kept.setdefault(path, []).append(line)
                else:
                    admitted.setdefault(path, []).append(line)
            else:
                kept.setdefault(path, []).append(line)
    consumer = {**old_consumer,
                "boundary": old_consumer["boundary"].split(", sql(pretty=True)")[0]
                + ", sql(pretty=True)<-execute, Plan<-execute",
                "test_execute_tpcds": {path: sorted(listed) for path, listed in sorted(kept.items())},
                "admitted": {path: sorted(listed) for path, listed in sorted(admitted.items())}}
    os.makedirs(out, exist_ok=True)
    dump(out, "fe_baseline.json.gz", baseline)
    dump(out, "fe_lines.json.gz", lines)
    dump(out, "fe_consumer.json.gz", consumer)
    print(json.dumps({"plan_digests": len(extra), "plan_cpu_total": round(sum(plan_cpu.values()), 3),
                      "worker_cpu_recorded": round(sum(planned["reports"]["worker"]["per"].get(str(index), 0.0)
                                                       for index in executed), 3),
                      "stage_cpu": planned["reports"]["worker"]["cpu"],
                      "consumer_before": sum(len(v) for v in outside.values()),
                      "consumer_after": sum(len(v) for v in kept.values()),
                      "admitted_by_file": {path: len(listed) for path, listed in sorted(admitted.items())},
                      "kept_as_first_task_only": first_only,
                      "plan_mask_lines": sum(len(table) for table in plan.values()),
                      "worker_mask_lines": [sum(len(t) for t in old_lines["worker"].values()),
                                            sum(len(t) for t in worker.values())],
                      "startup_lines": sum(len(v) for v in before_first_task.values())}))


def validate():
    """Replay the pinned tree against the maps next to this file; every verdict must be "ok"."""
    sys.path.insert(0, HERE)
    import fe_replay

    baseline = fe_replay._load("fe_baseline.json.gz")
    executed, everything = baseline["executed"], list(range(baseline["queries"]))
    failures = []

    def check(label, request, expected="ok"):
        began = time.monotonic()
        verdict = fe_replay._replay(request, 120)
        if verdict != expected:
            failures.append((label, verdict))
        return verdict, round(time.monotonic() - began, 2)

    print("environment", fe_replay._environment())
    print("everything", check("everything", {"sources": {}, "parent": True, "worker": everything, "plan": executed}))
    print("workers only", check("workers", {"sources": {}, "parent": False, "worker": executed, "plan": executed}))
    print("no plans", check("no plans", {"sources": {}, "parent": False, "worker": executed, "plan": []}))
    for index in executed:
        check(f"alone {index}", {"sources": {}, "parent": False, "worker": [index], "plan": [index]})
    back = executed[::-1][:7]
    print("subset", check("subset", {"sources": {}, "parent": False, "worker": back, "plan": back[:3]}))

    with open(os.path.join(ROOT, "sqlglot", "planner.py"), encoding="utf-8") as stream:
        planner = stream.read()
    same = planner.replace("        ctes = ctes or {}\n", "        ctes = ctes or dict()\n", 1)
    other = planner.replace('step.limit = int(limit.text("expression"))', 'step.limit = int(limit.text("expression")) + 1', 1)
    if same == planner or other == planner:
        failures.append(("controls", "the planner source does not have the expected lines"))
    request = {"parent": False, "worker": executed, "plan": executed}
    print("harmless edit", check("harmless edit", {**request, "sources": {"sqlglot/planner.py": same}}))
    verdict = fe_replay._replay({**request, "sources": {"sqlglot/planner.py": other}}, 120)
    print("limit edit", verdict)
    if not verdict.startswith("differs at plan"):
        failures.append(("limit edit", verdict))
    print("failures", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--record"]:
        record(sys.argv[2])
    elif sys.argv[1:2] == ["--startup"]:
        startup(*sys.argv[2:4])
    elif sys.argv[1:2] == ["--validate"]:
        sys.exit(validate())
    else:
        sys.path.insert(1, ROOT)
        build(sys.argv[1])
