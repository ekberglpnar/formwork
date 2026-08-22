"""Failures the sweep must not swallow."""

from __future__ import annotations

__all__ = ["QuotaWall"]


class QuotaWall(RuntimeError):
    """The provider is refusing everything, not throttling briefly.

    This one exception deliberately escapes ``run_arm``'s catch-all, because
    the sweep has to stop rather than record it as a run outcome. Dropped runs
    are not dropped evenly: arms that make more calls have more chances to hit
    the wall, so a sweep that grinds on past a hard quota quietly deletes the
    expensive arms' data and then reports what is left as a comparison.
    """
