import re
from textwrap import dedent

import pytest


def _ini(pytester: pytest.Pytester, with_asyncio: bool) -> tuple:
    if not with_asyncio:
        return ()
    pytest.importorskip("pytest_asyncio")
    pytester.makeini(
        """
        [pytest]
        asyncio_mode = auto
        asyncio_default_fixture_loop_scope = function
        """
    )
    return ("-p", "asyncio")


both_modes = pytest.mark.parametrize("with_asyncio", [False, True], ids=["plain", "pytest-asyncio"])


@both_modes
def test_async_fixture_setup_overlaps(pytester: pytest.Pytester, with_asyncio: bool):
    """Two grouped tests whose async fixture and body each sleep 1s finish together."""
    args = _ini(pytester, with_asyncio)
    pytester.makepyfile(
        dedent(
            """\
            import asyncio
            import pytest

            @pytest.fixture
            async def slow():
                await asyncio.sleep(1)
                return 1

            @pytest.fixture
            async def slow_gen():
                await asyncio.sleep(0.5)
                yield 2
                await asyncio.sleep(0.5)

            @pytest.mark.asyncio_concurrent(group="g")
            async def test_a(slow, slow_gen):
                await asyncio.sleep(1)
                assert (slow, slow_gen) == (1, 2)

            @pytest.mark.asyncio_concurrent(group="g")
            async def test_b(slow, slow_gen):
                await asyncio.sleep(1)
                assert (slow, slow_gen) == (1, 2)
            """
        )
    )
    result = pytester.runpytest_subprocess(*args)
    result.assert_outcomes(passed=2)
    # pytest's own session time, so interpreter startup under load doesn't count.
    match = re.search(r" in ([0-9.]+)s", result.stdout.lines[-1])
    assert match, result.stdout.lines[-1]
    # Serial setup/teardown: 2 x (1 + 0.5 + 0.5) + 1 = 5s. Overlapped: ~3s.
    assert float(match.group(1)) < 3.8


@both_modes
def test_dependency_chain_resolves_in_order(pytester: pytest.Pytester, with_asyncio: bool):
    """Deferred fixtures depending on deferred fixtures get the resolved values, set up
    dependencies first and torn down in reverse, while sync fixtures still work."""
    args = _ini(pytester, with_asyncio)
    pytester.makepyfile(
        dedent(
            """\
            import asyncio
            import pytest

            events = []

            @pytest.fixture
            async def base():
                events.append("base")
                yield 1
                print("EVENT base-down")

            @pytest.fixture
            async def mid(base):
                await asyncio.sleep(0.1)
                events.append("mid")
                yield base + 1
                print("EVENT mid-down")

            @pytest.fixture
            async def top(mid, request):
                events.append("top")
                return mid + 1

            @pytest.fixture
            async def other():
                await asyncio.sleep(0.1)
                return 1

            @pytest.fixture
            def sync_on_async(other):
                return other * 10

            @pytest.mark.asyncio_concurrent(group="g")
            async def test_chain(top, mid, request):
                assert top == 3
                assert mid == 2
                assert request.getfixturevalue("top") == 3
                assert events == ["base", "mid", "top"]

            @pytest.mark.asyncio_concurrent(group="g")
            async def test_sync_dependent(sync_on_async):
                assert sync_on_async == 10
            """
        )
    )
    result = pytester.runpytest(*args, "-s")
    result.assert_outcomes(passed=2)
    result.stdout.fnmatch_lines(["*EVENT mid-down*", "*EVENT base-down*"])


@both_modes
def test_failing_async_fixture_is_setup_error(pytester: pytest.Pytester, with_asyncio: bool):
    """A deferred fixture raising errors only the test that uses it, as a setup error."""
    args = _ini(pytester, with_asyncio)
    pytester.makepyfile(
        dedent(
            """\
            import asyncio
            import pytest

            @pytest.fixture
            async def broken():
                await asyncio.sleep(0.1)
                raise RuntimeError("fixture boom")

            @pytest.fixture
            async def fine():
                return 1

            @pytest.mark.asyncio_concurrent(group="g")
            async def test_broken(broken):
                pass

            @pytest.mark.asyncio_concurrent(group="g")
            async def test_fine(fine):
                assert fine == 1
            """
        )
    )
    result = pytester.runpytest(*args)
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(
        ["*ERROR at setup of test_broken*", "*RuntimeError: fixture boom*"]
    )


@both_modes
def test_asyncgen_teardown_runs_and_reports(pytester: pytest.Pytester, with_asyncio: bool):
    """Deferred async-generator teardown runs, and its error is that test's teardown error."""
    args = _ini(pytester, with_asyncio)
    pytester.makepyfile(
        dedent(
            """\
            import asyncio
            import pytest

            @pytest.fixture
            async def clean():
                yield 1
                await asyncio.sleep(0.1)
                print("EVENT clean-down")

            @pytest.fixture
            async def dirty():
                yield 2
                raise RuntimeError("teardown boom")

            @pytest.mark.asyncio_concurrent(group="g")
            async def test_clean(clean):
                pass

            @pytest.mark.asyncio_concurrent(group="g")
            async def test_dirty(dirty):
                pass
            """
        )
    )
    result = pytester.runpytest(*args, "-s")
    result.assert_outcomes(passed=2, errors=1)
    result.stdout.fnmatch_lines(
        [
            "*EVENT clean-down*",
            "*ERROR at teardown of test_dirty*",
            "*RuntimeError: teardown boom*",
        ]
    )
