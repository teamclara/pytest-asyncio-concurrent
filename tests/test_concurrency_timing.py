import time
from textwrap import dedent

import pytest

GROUPED_SLEEPS = dedent(
    """\
    import asyncio
    import pytest

    @pytest.mark.asyncio_concurrent(group="g")
    async def test_a():
        await asyncio.sleep(1)

    @pytest.mark.asyncio_concurrent(group="g")
    async def test_b():
        await asyncio.sleep(1)
    """
)


def _run_timed(pytester: pytest.Pytester, *args: str) -> float:
    pytester.makepyfile(GROUPED_SLEEPS)
    start = time.monotonic()
    result = pytester.runpytest_subprocess(*args)
    elapsed = time.monotonic() - start
    result.assert_outcomes(passed=2)
    return elapsed


def test_group_runs_concurrently(pytester: pytest.Pytester):
    """Two grouped 1s sleeps finish together, not one after the other."""
    assert _run_timed(pytester) < 1.9


def test_group_runs_concurrently_alongside_pytest_asyncio(pytester: pytest.Pytester):
    """With pytest-asyncio in auto mode, marked tests still run concurrently instead of
    being taken over and run one by one."""
    pytest.importorskip("pytest_asyncio")
    pytester.makeini(
        """
        [pytest]
        asyncio_mode = auto
        asyncio_default_fixture_loop_scope = function
        """
    )
    assert _run_timed(pytester, "-p", "asyncio") < 1.9
