import copy
import inspect
import asyncio
import contextlib
import functools

from typing import Any, Callable, Dict, Generator, List, Optional, Sequence
import warnings

import pytest
from _pytest import fixtures
from _pytest import nodes


# The running group's loop, so async fixtures share it with the group's tests.
# Python 3.14 removed the implicit loop `asyncio.get_event_loop()` used to create.
_group_loop: Optional[asyncio.AbstractEventLoop] = None


@contextlib.contextmanager
def group_event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """A fresh event loop for one concurrent group, closed when the group is done."""
    global _group_loop
    loop = asyncio.new_event_loop()
    _group_loop = loop
    try:
        yield loop
    finally:
        _group_loop = None
        loop.close()


# True while a concurrent group member's fixtures are being set up.
_deferring = False

_deferred_key = pytest.StashKey[List["Deferred"]]()


@contextlib.contextmanager
def deferring() -> Generator[None, None, None]:
    """Function-scoped async fixtures set up in this block return a `Deferred` instead of
    running, so the member's task can resolve them concurrently with its group."""
    global _deferring
    _deferring = True
    try:
        yield
    finally:
        _deferring = False


class Deferred:
    """A function-scoped async fixture whose setup was postponed to the call phase."""

    def __init__(self, func: Callable[..., Any], kwargs: Dict[str, Any], fixturedef: Any):
        self.func = func
        self.kwargs = kwargs
        self.fixturedef = fixturedef
        self.is_gen = inspect.isasyncgenfunction(func)
        self.gen: Any = None
        self.done = False
        self.value: Any = None
        self.exc: Optional[BaseException] = None
        # Resolved for a sync fixture during setup: torn down by pytest in normal order.
        self.eager = False
        self.torn_down = False
        self.teardown_exc: Optional[BaseException] = None

    async def resolve(self, order: Optional[List["Deferred"]] = None) -> Any:
        if self.exc is not None:
            raise self.exc
        if self.done:
            return self.value
        try:
            kwargs = {
                k: (await v.resolve(order) if isinstance(v, Deferred) else v)
                for k, v in self.kwargs.items()
            }
            if self.is_gen:
                self.gen = self.func(**kwargs)
                self.value = await self.gen.__anext__()
            else:
                self.value = await self.func(**kwargs)
        except BaseException as e:
            self.exc = e
            raise
        self.done = True
        # Later `request.getfixturevalue` calls see the value, not the placeholder.
        cached = self.fixturedef.cached_result
        if cached is not None:
            self.fixturedef.cached_result = (self.value, *cached[1:])
        if order is not None:
            order.append(self)
        return self.value

    def resolve_eager(self) -> Any:
        self.eager = True
        for v in self.kwargs.values():
            if isinstance(v, Deferred):
                v.resolve_eager()
        assert _group_loop is not None
        return _group_loop.run_until_complete(self.resolve())

    async def teardown(self) -> None:
        if not self.is_gen or self.gen is None or self.torn_down:
            return
        self.torn_down = True
        try:
            await self.gen.__anext__()
        except StopAsyncIteration:
            return
        except BaseException as e:
            self.teardown_exc = e
            return
        self.teardown_exc = ValueError("Async generator fixture didn't stop. Yield only once.")

    def finish(self) -> None:
        """pytest finalizer: tears down what the concurrent pass didn't, reports its error."""
        if self.is_gen and self.gen is not None and not self.torn_down:
            assert _group_loop is not None
            _group_loop.run_until_complete(self.teardown())
        if self.teardown_exc is not None:
            raise self.teardown_exc


async def resolve_deferred(item: pytest.Function) -> bool:
    """Resolve the item's deferred fixtures in dependency order; False if it had none."""
    order: List[Deferred] = []
    for name in item.fixturenames:
        value = item.funcargs.get(name)
        if isinstance(value, Deferred):
            item.funcargs[name] = await value.resolve(order)
    item.stash[_deferred_key] = order
    return bool(order)


async def teardown_deferred(item: pytest.Function) -> None:
    for d in reversed(item.stash.get(_deferred_key, [])):
        if not d.eager:
            await d.teardown()


def has_deferred(item: pytest.Function) -> bool:
    return any(isinstance(v, Deferred) for v in item.funcargs.values())


@pytest.hookimpl(specname="pytest_fixture_setup", wrapper=True, tryfirst=True)
def pytest_fixture_setup_defer_async(
    fixturedef: pytest.FixtureDef, request: pytest.FixtureRequest
) -> Generator[None, Any, Any]:
    # tryfirst wrapper: outermost, so pytest-asyncio never sees the async func and runs it.
    func = fixturedef.func
    if not _deferring or fixturedef.scope != "function":
        return (yield)

    if inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):
        if inspect.isasyncgenfunction(func):

            def _deferred_gen(**kwargs: Any) -> Any:
                d = Deferred(func, kwargs, fixturedef)
                yield d
                d.finish()

            placeholder: Callable[..., Any] = _deferred_gen
        else:

            def _deferred_func(**kwargs: Any) -> Any:
                return Deferred(func, kwargs, fixturedef)

            placeholder = _deferred_func

        fixturedef.func = placeholder  # type: ignore[misc]
        try:
            return (yield)
        finally:
            fixturedef.func = func  # type: ignore[misc]

    # A sync fixture can't await: resolve the deferred fixtures it depends on now.
    for argname in fixturedef.argnames:
        value = request.getfixturevalue(argname)
        if isinstance(value, Deferred):
            value.resolve_eager()
    return (yield)


@pytest.hookimpl(specname="pytest_fixture_setup", tryfirst=True)
def pytest_fixture_setup_wrap_async(
    fixturedef: pytest.FixtureDef, request: pytest.FixtureRequest
) -> None:
    _wrap_async_fixture(fixturedef)


def _wrap_async_fixture(fixturedef: pytest.FixtureDef) -> None:
    """Wraps the fixture function of an async fixture in a synchronous function."""
    if inspect.isasyncgenfunction(fixturedef.func):
        _wrap_asyncgen_fixture(fixturedef)
    elif inspect.iscoroutinefunction(fixturedef.func):
        _wrap_asyncfunc_fixture(fixturedef)


def _wrap_asyncgen_fixture(fixturedef: pytest.FixtureDef) -> None:
    fixtureFunc = fixturedef.func

    @functools.wraps(fixtureFunc)
    def _asyncgen_fixture_wrapper(**kwargs: Any):
        event_loop = asyncio.new_event_loop()
        gen_obj = fixtureFunc(**kwargs)

        async def setup():
            res = await gen_obj.__anext__()  # type: ignore[union-attr]
            return res

        async def teardown() -> None:
            try:
                await gen_obj.__anext__()  # type: ignore[union-attr]
            except StopAsyncIteration:
                pass
            else:
                msg = "Async generator fixture didn't stop."
                msg += "Yield only once."
                raise ValueError(msg)

        result = event_loop.run_until_complete(setup())
        yield result
        event_loop.run_until_complete(teardown())

    fixturedef.func = _asyncgen_fixture_wrapper  # type: ignore[misc]


def _wrap_asyncfunc_fixture(fixturedef: pytest.FixtureDef) -> None:
    fixtureFunc = fixturedef.func

    @functools.wraps(fixtureFunc)
    def _async_fixture_wrapper(**kwargs: Dict[str, Any]):
        async def setup():
            res = await fixtureFunc(**kwargs)
            return res

        if _group_loop is not None:
            return _group_loop.run_until_complete(setup())
        event_loop = asyncio.new_event_loop()
        try:
            return event_loop.run_until_complete(setup())
        finally:
            event_loop.close()

    fixturedef.func = _async_fixture_wrapper  # type: ignore[misc]


fixture_cache_key = pytest.StashKey[Dict[str, Optional[Sequence[pytest.FixtureDef[Any]]]]]()


@pytest.hookimpl(specname="pytest_sessionstart", trylast=True)
def pytest_sessionstart_cache_fixture(session: pytest.Session):
    # This function in general utilized some private properties.

    # This is to solve two problem:
    # 1. Funtion scoped fixture result value got shared in different tests in same group.
    # 2. And fixture teardown got registered under right test using it.

    # FixtureDef for each fixture is unique and held in FixtureManger and got injected into
    # pytest.Item when the Item is constructed, and FixtureDef class is also in charge of
    # holding finalizers and cache value.

    # Fixture value caching is highly coupled with pytest entire lifecycle, implementing a
    # thirdparty fixture cache manager will be hard.
    # The first problem can be solved by shallow copy the fixtureDef, to split the cache_value.
    # The finalizers are stored in a private list property in fixtureDef, which need to touch
    # private API anyway.

    # If the private API change, finalizer errors from this fixture but in different
    # tests in same group will be reported in one function.

    fixManager: fixtures.FixtureManager = session.config.pluginmanager.get_plugin(
        "funcmanage"
    )  # type: ignore
    getfixturedefs_original = fixManager.getfixturedefs

    @functools.wraps(getfixturedefs_original)
    def getfixturedefs_wrapper(
        argname: str,
        node: nodes.Node,
    ) -> Optional[Sequence[pytest.FixtureDef[Any]]]:
        if fixture_cache_key not in node.stash:
            node.stash[fixture_cache_key] = {}

        cache = node.stash[fixture_cache_key]
        if argname not in cache:
            fixtureDefs = getfixturedefs_original(argname, node)

            if fixtureDefs:
                fixtureDefs = tuple(_clone_function_fixture(fixDef) for fixDef in fixtureDefs)

            cache[argname] = fixtureDefs

        return cache[argname]

    fixManager.getfixturedefs = getfixturedefs_wrapper  # type: ignore


def _clone_function_fixture(fixtureDef: pytest.FixtureDef) -> pytest.FixtureDef:
    if fixtureDef.scope != "function":
        return fixtureDef

    new_fixdef = copy.copy(fixtureDef)
    if hasattr(fixtureDef, "_finalizers"):
        new_fixdef._finalizers = []  # type: ignore
    else:
        warnings.warn(
            f"""
            pytest {pytest.__version__} changed internal property which this plugin relies on.
            The teardown error in fixture {fixtureDef.argname} might be reported in wrong place.
            Please raise an issue.
            """
        )

    return new_fixdef
