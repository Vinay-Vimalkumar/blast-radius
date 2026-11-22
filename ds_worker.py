import hashlib
import sys
import unittest
from pathlib import Path


def main():
    import duckdb
    import numpy
    import pandas
    from tests.test_executor import TestExecutor

    if (duckdb.__version__,numpy.__version__,pandas.__version__) != ("2.0.0","2.5.5","3.1.1"):
        return 1
    stage, target, digest, *indices = sys.argv[1:]
    stage = Path(stage)
    original = TestExecutor.setUpClass
    indices = list(map(int,indices))

    def setup(cls):
        original()
        rows = cls.tpcds_sqls
        if len(rows) != 99 or any(not rows[i][0].get("execute") for i in indices):
            raise ValueError("fixtures")
        cls.tpcds_sqls = [rows[i] for i in indices]

    TestExecutor.setUpClass = classmethod(setup)

    class Result(unittest.TestResult):
        successes = 0
        subtests = 0

        def addSuccess(self, test):
            self.successes += 1
            super().addSuccess(test)

        def addSubTest(self,test,subtest,err):
            if err is None:
                self.subtests += 1
            super().addSubTest(test,subtest,err)

    report = unittest.TestSuite([TestExecutor("test_execute_tpcds")]).run(Result())
    if not (report.wasSuccessful() and report.testsRun == report.successes == 1
            and report.subtests == len(indices) and not report.skipped
            and not report.expectedFailures and not report.unexpectedSuccesses):
        return 1
    module = sys.modules.get(target.removesuffix(".py").replace("/",".").removesuffix(".__init__"))
    if not module or Path(module.__file__) != stage / target or hashlib.sha256((stage/target).read_bytes()).hexdigest() != digest:
        return 1
    origins = list(stage.glob("worker-*"))
    if len(origins) != 1 or int(origins[0].read_text()) != len(indices):
        return 1
    print("verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
