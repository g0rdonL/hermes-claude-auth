# pyright: reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false
"""Tests for the API-credits fallback (v1.8.0).

When a request billed to the Claude subscription returns HTTP 429 carrying
the unified subscription-window headers, the call must be transparently
retried against an ``x-api-key`` credits client.  Headerless
third-party-detection 429s must never trigger the fallback, clients that
already use ``x-api-key`` must be left alone, and a failed credits retry
must re-raise the original 429 so the auto-wait still applies.
"""

import sys
import types

import pytest

import anthropic_billing_bypass as bp


class _FakeResponse:
    def __init__(self, headers):
        self.headers = headers
        self.status_code = 429


class _Fake429(Exception):
    def __init__(self, headers):
        super().__init__("Error code: 429 - rate_limit_error")
        self.status_code = 429
        self.response = _FakeResponse(headers)


def _quota_429():
    return _Fake429(
        {
            "anthropic-ratelimit-unified-5h-status": "throttled",
            "anthropic-ratelimit-unified-5h-reset": str(int(__import__("time").time()) + 600),
        }
    )


def _headerless_429():
    return _Fake429({})


class _FakeMessages:
    """Stand-in for ``anthropic.resources.messages.Messages``.

    ``create_fn`` / ``stream_fn`` override per-instance behaviour so tests can
    make the OAuth-side call fail while the fallback-side call succeeds.
    """

    def __init__(self, client=None):
        self._client = client
        self.create_fn = None
        self.stream_fn = None

    def create(self, **kwargs):
        if self.create_fn is not None:
            return self.create_fn(**kwargs)
        return ("create", kwargs)

    def stream(self, **kwargs):
        if self.stream_fn is not None:
            return self.stream_fn(**kwargs)
        return ("stream", kwargs)


class _FakeAnthropic:
    def __init__(self, api_key=None, auth_token=None, base_url=None):
        self.api_key = api_key
        self.auth_token = auth_token
        self.base_url = base_url
        self._client = types.SimpleNamespace(headers={})
        self.messages = _FakeMessages(client=self)


@pytest.fixture
def fake_sdk(monkeypatch):
    """Install a fake ``anthropic`` package and reset all fallback state."""
    mod = types.ModuleType("anthropic")
    resources = types.ModuleType("anthropic.resources")
    messages_mod = types.ModuleType("anthropic.resources.messages")
    messages_mod.Messages = _FakeMessages
    resources.messages = messages_mod
    mod.resources = resources
    mod.Anthropic = _FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.setitem(sys.modules, "anthropic.resources", resources)
    monkeypatch.setitem(sys.modules, "anthropic.resources.messages", messages_mod)

    saved = (_FakeMessages.create, _FakeMessages.stream)
    monkeypatch.setattr(bp, "_FB_STATE", {"until": 0.0, "client": None, "client_failed": False})
    monkeypatch.setattr(bp, "_FB_ORIG", {})

    assert bp.install_api_credits_fallback() is True

    yield types.SimpleNamespace(mod=mod, restore=saved)

    # Un-patch the class so other tests see pristine methods.
    _FakeMessages.create, _FakeMessages.stream = saved
    if getattr(_FakeMessages, "_API_CREDITS_FALLBACK_PATCHED", False):
        delattr(_FakeMessages, "_API_CREDITS_FALLBACK_PATCHED")


@pytest.fixture
def env(fake_sdk, monkeypatch):
    monkeypatch.setenv("HERMES_API_FALLBACK_API_KEY", "sk-credits")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return fake_sdk


def _oauth_client():
    return _FakeAnthropic(auth_token="oauth-access-token")


# ---------------------------------------------------------------------------
# Fallback triggers on a genuine subscription-quota 429
# ---------------------------------------------------------------------------


def test_quota_429_retries_with_credits(env):
    client = _oauth_client()
    client.messages.create_fn = lambda **kw: (_ for _ in ()).throw(_quota_429())

    result = client.messages.create(model="claude-opus-4-6", messages=[{"role": "user", "content": "hi"}])

    assert result == ("create", {"model": "claude-opus-4-6", "messages": [{"role": "user", "content": "hi"}]})
    # The fallback client was built from the credits key.
    assert bp._FB_STATE["client"] is not None
    assert bp._FB_STATE["client"].api_key == "sk-credits"


def test_stream_quota_429_retries_with_credits_stream(env):
    client = _oauth_client()
    client.messages.stream_fn = lambda **kw: (_ for _ in ()).throw(_quota_429())

    result = client.messages.stream(model="claude-opus-4-6", messages=[])

    assert result[0] == "stream"


def test_exhausted_window_routes_preemptively(env):
    client = _oauth_client()
    calls = []

    def oauth_create(**kw):
        calls.append("oauth")
        raise _quota_429()

    client.messages.create_fn = oauth_create
    client.messages.create(model="m", messages=[])

    # Second call: window still exhausted — OAuth side must not be hit again.
    client.messages.create_fn = lambda **kw: (_ for _ in ()).throw(
        AssertionError("OAuth call should have been skipped"))
    result = client.messages.create(model="m", messages=[])
    assert result[0] == "create"
    assert calls == ["oauth"]
    assert bp._FB_STATE["until"] > bp.time.time()


def test_oauth_side_untouched_when_window_reset(env, monkeypatch):
    client = _oauth_client()
    client.messages.create_fn = lambda **kw: (_ for _ in ()).throw(_quota_429())
    client.messages.create(model="m", messages=[])
    assert bp._FB_STATE["until"] > bp.time.time()

    # Window has reset → subscription is tried again.
    monkeypatch.setattr(bp.time, "time", lambda: bp._FB_STATE["until"] + 1)
    seen = []
    client.messages.create_fn = lambda **kw: seen.append(kw) or ("oauth-ok", kw)
    result = client.messages.create(model="m", messages=[])
    assert result[0] == "oauth-ok"
    assert seen


# ---------------------------------------------------------------------------
# Kwargs cleaning
# ---------------------------------------------------------------------------


def test_fallback_strips_oauth_only_decorations(env):
    client = _oauth_client()
    client.messages.create_fn = lambda **kw: (_ for _ in ()).throw(_quota_429())
    kwargs = {
        "model": "claude-opus-4-6",
        "messages": [{"role": "user", "content": "hi"}],
        "system": [
            {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.217.a1b;"},
            {"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."},
            {"type": "text", "text": "real system guidance"},
        ],
        "metadata": {"user_id": '{"device_id":"x"}', "trace": "keep-me"},
        "extra_headers": {"user-agent": "claude-cli/2.1.217 (external, sdk-cli)", "x-app": "cli"},
        "extra_query": {"beta": "true"},
        "extra_body": {"context_management": {"edits": []}},
        "max_tokens": 32000,
    }

    _, clean = client.messages.create(**kwargs)

    system_texts = [e.get("text") for e in clean["system"]]
    assert not any(t.startswith("x-anthropic-billing-header") for t in system_texts)
    assert "You are Claude Code, Anthropic's official CLI for Claude." in system_texts
    assert "real system guidance" in system_texts
    assert clean["metadata"] == {"trace": "keep-me"}
    for key in ("extra_headers", "extra_query", "extra_body"):
        assert key not in clean
    assert clean["max_tokens"] == 32000


def test_original_kwargs_not_mutated(env):
    client = _oauth_client()
    client.messages.create_fn = lambda **kw: (_ for _ in ()).throw(_quota_429())
    system = [
        {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.217.a1b;"},
        {"type": "text", "text": "identity"},
    ]
    kwargs = {"model": "m", "messages": [], "system": system, "extra_headers": {"x-app": "cli"}}

    client.messages.create(**kwargs)

    assert len(kwargs["system"]) == 2  # billing header still present for the OAuth retry
    assert kwargs["extra_headers"] == {"x-app": "cli"}


# ---------------------------------------------------------------------------
# Non-fallback cases
# ---------------------------------------------------------------------------


def test_headerless_429_never_falls_back(env):
    client = _oauth_client()
    client.messages.create_fn = lambda **kw: (_ for _ in ()).throw(_headerless_429())

    with pytest.raises(_Fake429):
        client.messages.create(model="m", messages=[])

    assert bp._FB_STATE["client"] is None  # fallback client never built
    assert bp._FB_STATE["until"] == 0.0


def test_non_429_error_propagates(env):
    client = _oauth_client()
    client.messages.create_fn = lambda **kw: (_ for _ in ()).throw(ValueError("boom"))

    with pytest.raises(ValueError):
        client.messages.create(model="m", messages=[])

    assert bp._FB_STATE["client"] is None


def test_api_key_primary_client_never_redirected(env):
    client = _FakeAnthropic(api_key="sk-primary")
    client.messages.create_fn = lambda **kw: (_ for _ in ()).throw(_quota_429())

    with pytest.raises(_Fake429):
        client.messages.create(model="m", messages=[])

    assert bp._FB_STATE["client"] is None


def test_disabled_via_env(env, monkeypatch):
    monkeypatch.setenv("HERMES_API_FALLBACK", "0")
    client = _oauth_client()
    client.messages.create_fn = lambda **kw: (_ for _ in ()).throw(_quota_429())

    with pytest.raises(_Fake429):
        client.messages.create(model="m", messages=[])

    assert bp._FB_STATE["client"] is None


def test_no_key_configured(fake_sdk, monkeypatch):
    monkeypatch.delenv("HERMES_API_FALLBACK_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = _oauth_client()
    client.messages.create_fn = lambda **kw: (_ for _ in ()).throw(_quota_429())

    with pytest.raises(_Fake429):
        client.messages.create(model="m", messages=[])

    assert bp._FB_STATE["client"] is None


def test_anthropic_api_key_env_used_as_fallback_key(fake_sdk, monkeypatch):
    monkeypatch.delenv("HERMES_API_FALLBACK_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    client = _oauth_client()
    client.messages.create_fn = lambda **kw: (_ for _ in ()).throw(_quota_429())

    client.messages.create(model="m", messages=[])

    assert bp._FB_STATE["client"].api_key == "sk-from-env"


def test_custom_base_url(env, monkeypatch):
    monkeypatch.setenv("HERMES_API_FALLBACK_BASE_URL", "https://proxy.example.com")
    client = _oauth_client()
    client.messages.create_fn = lambda **kw: (_ for _ in ()).throw(_quota_429())

    client.messages.create(model="m", messages=[])

    assert bp._FB_STATE["client"].base_url == "https://proxy.example.com"


# ---------------------------------------------------------------------------
# Credits failure → original 429 propagates (auto-wait can still apply)
# ---------------------------------------------------------------------------


def test_failed_credits_retry_reraises_original_429(env):
    client = _oauth_client()

    def oauth_create(**kw):
        raise _quota_429()

    client.messages.create_fn = oauth_create
    # Force the fallback client to fail too.
    bp._FB_STATE["client"] = _FakeAnthropic(api_key="sk-credits")
    bp._FB_STATE["client"].messages.create_fn = lambda **kw: (_ for _ in ()).throw(
        ValueError("credits exhausted"))

    with pytest.raises(_Fake429):
        client.messages.create(model="m", messages=[])

    # Exhaustion was still recorded (auto-wait/preemptive logic intact).
    assert bp._FB_STATE["until"] > bp.time.time()


def test_install_is_idempotent(env):
    assert bp.install_api_credits_fallback() is True
    # Still the same wrapper, originals preserved.
    assert _FakeMessages.create.__wrapped__ is bp._FB_ORIG["create"]
