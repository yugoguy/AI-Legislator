"""Sandboxed Python execution in a separate process.

Source
------
AI Scientist v2 `treesearch/interpreter.py` (itself derived from AIDE). The
process/queue execution model, stdout/stderr capture, SIGINT-then-kill timeout,
and exception summarization are carried over essentially unchanged.

Reused (near-verbatim)
----------------------
- Interpreter: a child process holding a persistent global scope, three queues
  (code in, output out, events out), SIGINT-then-kill timeout handling, and
  EOF-marker draining of stdout/stderr.
- exception_summary: traceback formatting + structured exc info / stack.

Changed
-------
- ExecutionResult is a plain dataclass exposing asdict() instead of depending on
  dataclasses_json (one fewer dependency; still JSON-dumpable).
- `humanize` dependency removed; timeout / timing messages report plain seconds.
- `shutup` warning-muting is now best-effort (optional import).
- Framework-path traceback filtering ("treesearch/") removed, since this
  package's path differs. The agent file path is still collapsed to its bare
  filename, which is what keeps tracebacks readable.
"""

from __future__ import annotations

import logging
import os
import queue
import signal
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from multiprocessing import Process, Queue
from pathlib import Path

logger = logging.getLogger("ai-legislator")


@dataclass
class ExecutionResult:
    """Output and metadata from executing one code snippet."""

    term_out: list[str]
    exec_time: float
    exc_type: str | None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def exception_summary(e, working_dir, exec_file_name):
    """Summarize an exception + stack trace, collapsing the workspace path."""
    tb_lines = traceback.format_exception(e)
    tb_str = "".join(l for l in tb_lines if "importlib" not in l)
    # Replace the absolute path to the agent's file with just its filename.
    tb_str = tb_str.replace(str(working_dir / exec_file_name), exec_file_name)

    exc_info = {}
    if hasattr(e, "args"):
        exc_info["args"] = [str(i) for i in e.args]
    for att in ["name", "msg", "obj"]:
        if hasattr(e, att):
            exc_info[att] = str(getattr(e, att))

    tb = traceback.extract_tb(e.__traceback__)
    exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]

    return tb_str, e.__class__.__name__, exc_info, exc_stack


class RedirectQueue:
    """File-like object that pushes writes onto a multiprocessing Queue."""

    def __init__(self, q):
        self.queue = q

    def write(self, msg):
        self.queue.put(msg)

    def flush(self):
        pass


class Interpreter:
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 3600,
        agent_file_name: str = "runfile.py",
        env_vars: dict[str, str] | None = None,
    ):
        """A standalone Python REPL (in a child process) with a time limit.

        Args:
            working_dir: working directory for the executed code.
            timeout: per-execution wall-clock limit, in seconds.
            agent_file_name: filename the code is written to before exec.
            env_vars: environment variables to set inside the child process.
        """
        self.working_dir = Path(working_dir).resolve()
        assert self.working_dir.exists(), f"Working dir {self.working_dir} does not exist"
        self.timeout = timeout
        self.agent_file_name = agent_file_name
        self.env_vars = env_vars or {}
        self.process: Process | None = None

    def child_proc_setup(self, result_outq: Queue) -> None:
        try:  # best-effort warning muting; absence is non-fatal
            import shutup

            shutup.mute_warnings()
        except ImportError:
            pass

        for key, value in self.env_vars.items():
            os.environ[key] = value

        os.chdir(str(self.working_dir))
        sys.path.append(str(self.working_dir))
        sys.stdout = sys.stderr = RedirectQueue(result_outq)

    def _run_session(self, code_inq: Queue, result_outq: Queue, event_outq: Queue) -> None:
        self.child_proc_setup(result_outq)

        global_scope: dict = {}
        while True:
            code = code_inq.get()
            os.chdir(str(self.working_dir))
            with open(self.agent_file_name, "w") as f:
                f.write(code)

            event_outq.put(("state:ready",))
            try:
                exec(compile(code, self.agent_file_name, "exec"), global_scope)
            except BaseException as e:
                tb_str, e_cls_name, exc_info, exc_stack = exception_summary(
                    e, self.working_dir, self.agent_file_name
                )
                result_outq.put(tb_str)
                if e_cls_name == "KeyboardInterrupt":
                    e_cls_name = "TimeoutError"
                event_outq.put(("state:finished", e_cls_name, exc_info, exc_stack))
            else:
                event_outq.put(("state:finished", None, None, None))

            result_outq.put("<|EOF|>")

    def create_process(self) -> None:
        # Three queues: code to child, stdout/stderr from child, lifecycle events.
        self.code_inq, self.result_outq, self.event_outq = Queue(), Queue(), Queue()
        self.process = Process(
            target=self._run_session,
            args=(self.code_inq, self.result_outq, self.event_outq),
        )
        self.process.start()

    def _drain_queues(self):
        """Quickly drain all in-flight messages to prevent blocking."""
        for q in (self.result_outq, self.event_outq, self.code_inq):
            while not q.empty():
                try:
                    q.get_nowait()
                except Exception:
                    break

    def cleanup_session(self):
        if self.process is None:
            return
        self.process.terminate()
        self._drain_queues()
        self.process.join(timeout=2)
        if self.process.exitcode is None:
            logger.warning("Child process failed to terminate gracefully, killing it..")
            self.process.kill()
            self._drain_queues()
            self.process.join(timeout=2)
        self.process.close()
        self.process = None

    def run(self, code: str, reset_session: bool = True) -> ExecutionResult:
        """Execute `code` in the child process and return its captured output."""
        logger.debug(f"REPL is executing code (reset_session={reset_session})")

        if reset_session:
            if self.process is not None:
                self.cleanup_session()
            self.create_process()
        else:
            assert self.process is not None  # must be True on first exec

        assert self.process.is_alive()
        self.code_inq.put(code)

        # Wait for the child to actually start (don't interrupt its setup).
        try:
            state = self.event_outq.get(timeout=10)
        except queue.Empty:
            msg = "REPL child process failed to start execution"
            logger.critical(msg)
            while not self.result_outq.empty():
                logger.error(f"REPL output queue dump: {self.result_outq.get()}")
            raise RuntimeError(msg) from None
        assert state[0] == "state:ready", state
        start_time = time.time()

        child_in_overtime = False
        while True:
            try:
                state = self.event_outq.get(timeout=1)
                assert state[0] == "state:finished", state
                exec_time = time.time() - start_time
                break
            except queue.Empty:
                if not child_in_overtime and not self.process.is_alive():
                    msg = "REPL child process died unexpectedly"
                    logger.critical(msg)
                    while not self.result_outq.empty():
                        logger.error(f"REPL output queue dump: {self.result_outq.get()}")
                    raise RuntimeError(msg) from None

                if self.timeout is None:
                    continue
                running_time = time.time() - start_time
                if running_time > self.timeout:
                    assert reset_session, "Timeout occurred in interactive session"
                    os.kill(self.process.pid, signal.SIGINT)
                    child_in_overtime = True
                    if running_time > self.timeout + 60:
                        logger.warning("Child failed to terminate, killing it..")
                        self.cleanup_session()
                        state = (None, "TimeoutError", {}, [])
                        exec_time = self.timeout
                        break

        # Read all stdout/stderr up to the EOF marker. Emptiness alone is not a
        # stop condition because the child's feeder thread may still be flushing.
        output: list[str] = []
        while not self.result_outq.empty() or not output or output[-1] != "<|EOF|>":
            output.append(self.result_outq.get())
        output.pop()  # remove the EOF marker

        e_cls_name, exc_info, exc_stack = state[1:]
        if e_cls_name == "TimeoutError":
            output.append(f"TimeoutError: execution exceeded the {self.timeout}s limit.")
        else:
            output.append(f"Execution time: {exec_time:.2f}s (limit {self.timeout}s).")

        return ExecutionResult(output, exec_time, e_cls_name, exc_info, exc_stack)
