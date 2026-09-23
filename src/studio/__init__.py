"""EduBehaviors Studio."""

__all__ = ["create_app"]


def create_app(*args, **kwargs):
    from .app import create_app as _create

    return _create(*args, **kwargs)
