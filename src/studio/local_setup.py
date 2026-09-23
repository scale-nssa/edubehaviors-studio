"""The two pages that are the whole of "set this thing up": Models and Datasets.

The hosted app never needed either: an operator chose the models and shipped
the corpora out of band. Locally the researcher does both, so they live in the
nav where they can be found without being told they exist.

Also home to the guards the construct routes call before anything costs money:
`models_ready_or_redirect` and `preflight` (docs/local-mode-plan.md §4.2, §4.4).
"""

from __future__ import annotations

import threading

from flask import (
    Blueprint,
    abort,
    flash,
    g,
    redirect,
    render_template,
    request,
    url_for,
)

from . import annotate, model_settings, spend
from .providers import PROVIDERS, SUGGESTED

bp = Blueprint("setup", __name__)

# role -> (ok, message) from the last Test connection, for this process only.
_last_test: dict[str, tuple[bool, str]] = {}
_last_test_lock = threading.Lock()


def _require_user():
    if not g.get("reviewer"):
        abort(redirect(url_for("start")))


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

@bp.route("/models")
def models():
    _require_user()
    roles = []
    for role in model_settings.ROLES:
        spec = model_settings.role_spec(role)
        nick = model_settings.role_nick(role)
        with _last_test_lock:
            last = _last_test.get(role)
        roles.append({
            "role": role,
            "title": model_settings.ROLE_TITLES[role],
            "spec": spec,
            "nick": nick,
            "verified": model_settings.is_verified(role),
            "price": model_settings.price(nick),
            "last": last,
        })
    keys = [
        {
            "provider": p,
            "shown": model_settings.redact_key(model_settings.key_for(p.id)),
            "used": any(r["spec"]["provider"] == p.id for r in roles),
        }
        for p in PROVIDERS.values() if p.key_env
    ]
    return render_template(
        "models.html",
        roles=roles,
        keys=keys,
        providers=PROVIDERS,
        suggested=SUGGESTED,
        same_model=model_settings.same_annotator_model(),
        busy=annotate.busy(),
        cap=model_settings.spend_cap(),
        lifetime=spend.lifetime_usd(),
        session=spend.session_usd(),
        unpriced=spend.unpriced_calls(),
        data_dir=model_settings.data_dir(),
        next_url=request.args.get("next") or "",
    )


@bp.route("/models/key/<provider>", methods=["POST"])
def save_key(provider: str):
    _require_user()
    if provider not in PROVIDERS or not PROVIDERS[provider].key_env:
        abort(404)
    model_settings.set_key(provider, request.form.get("key") or "")
    # A new key invalidates every role on that provider — the fingerprint
    # carries a hash of the key — so they must be re-tested. Say so.
    flash(f"{PROVIDERS[provider].title} key saved. Test each role that uses it.")
    return redirect(url_for("setup.models") + "#roles")


@bp.route("/models/role/<role>", methods=["POST"])
def save_role(role: str):
    _require_user()
    if role not in model_settings.ROLES:
        abort(404)
    if annotate.busy():
        # A rebind mid-round would have half the calls land under one
        # nickname and half under another.
        flash("A round is annotating right now. Change models once it finishes.")
        return redirect(url_for("setup.models") + f"#{role}")
    spec = {k: request.form.get(k) for k in model_settings.SPEC_FIELDS}
    try:
        model_settings.set_role(role, spec)
    except ValueError as exc:
        flash(str(exc))
        return redirect(url_for("setup.models") + f"#{role}")
    with _last_test_lock:
        _last_test.pop(role, None)
    if request.form.get("then_test"):
        return _test(role)
    flash(f"{model_settings.ROLE_TITLES[role]} saved. Press Test connection to use it.")
    return redirect(url_for("setup.models") + f"#{role}")


@bp.route("/models/role/<role>/test", methods=["POST"])
def test_role(role: str):
    _require_user()
    if role not in model_settings.ROLES:
        abort(404)
    return _test(role)


def _test(role: str):
    from .llm import test_role as run_test

    ok, message = run_test(role)
    with _last_test_lock:
        _last_test[role] = (ok, message)
    return redirect(url_for("setup.models") + f"#{role}")


@bp.route("/models/cap", methods=["POST"])
def save_cap():
    _require_user()
    try:
        model_settings.set_spend_cap(float(request.form.get("cap") or 0))
    except ValueError as exc:
        flash(f"Not a valid cap: {exc}")
    return redirect(url_for("setup.models") + "#spend")


# --------------------------------------------------------------------------- #
# Guards for paid actions
# --------------------------------------------------------------------------- #

def models_ready_or_redirect():
    """None when every role is verified; otherwise a redirect to /models."""
    if model_settings.ready():
        return None
    missing = ", ".join(
        model_settings.ROLE_TITLES[r] for r in model_settings.unverified_roles()
    )
    flash(f"Set up your models first — not yet verified: {missing}.")
    return redirect(url_for("setup.models", next=request.referrer or ""))


def preflight(estimate, *, title: str, fields: dict | None = None,
              back: str | None = None):
    """Show the estimate and ask, unless the form already said yes.

    Returns None when the caller may go ahead, or a response to return. Every
    paid action passes through here: nothing spends the user's money without
    their having seen roughly how much (plan §4.4). An action the cache
    covers entirely (zero calls) goes straight through — there is nothing to
    confirm — but the cap still applies to anything that does cost.
    """
    if estimate.calls == 0:
        return None
    # Blocked on this action's own estimate, not the chained upper bound: a
    # chain may not happen, and `spend.check` stops it per call if it does.
    if spend.would_exceed(estimate.usd):
        return render_template(
            "confirm_spend.html", est=estimate, title=title, blocked=True,
            cap=model_settings.spend_cap(), lifetime=spend.lifetime_usd(),
            back=back, fields={}, action=request.path,
        ), 402
    if request.form.get("confirmed") == "1":
        return None
    posted = {k: v for k, v in request.form.items() if k != "confirmed"}
    posted.update(fields or {})
    return render_template(
        "confirm_spend.html", est=estimate, title=title, blocked=False,
        cap=model_settings.spend_cap(), lifetime=spend.lifetime_usd(),
        back=back, fields=posted, action=request.path,
    )


@bp.app_context_processor
def _setup_state():
    # The nav shows a dot on Models until every role is verified, and the
    # running spend estimate, so neither is ever more than a glance away.
    try:
        return {
            "models_ready": model_settings.ready(),
            "spend_lifetime": spend.lifetime_usd(),
        }
    except Exception:  # noqa: BLE001 — a nav badge must never break a page
        return {"models_ready": True, "spend_lifetime": 0.0}


def register(app) -> None:
    app.register_blueprint(bp)
    # Datasets live in their own module; imported here so app.py gains one line.
    from . import datasets_page

    app.register_blueprint(datasets_page.bp)
