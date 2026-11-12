"""Split a multi-document upload into per-document page groups.

Classifies every page (cheap: one image, small context) and groups
consecutive pages predicted as the same doc_type into one sub-document. A
type change is the boundary - this is a heuristic, not a layout model, and it
works because a scanned batch is a run of pages per document, not an
interleaving of them.
"""

from typing import NamedTuple

from . import pipeline
from .doc import Page


class Segment(NamedTuple):
    pages: list[Page]
    cls: dict  # pipeline.classify()'s output for this group's first page


def split(pages: list[Page]) -> list[Segment]:
    """One classify call per page, then group consecutive same-type runs.

    A single-page upload - the common case - costs exactly the one classify
    call pipeline.run_pages would have made anyway; run_pages is handed the
    result via Segment.cls so it isn't classified twice.
    """
    per_page = [pipeline.classify([p]) for p in pages]

    segments = []
    group, group_cls = [pages[0]], per_page[0]
    for page, cls in zip(pages[1:], per_page[1:]):
        if cls["doc_type"] == group_cls["doc_type"]:
            group.append(page)
        else:
            segments.append(Segment(group, group_cls))
            group, group_cls = [page], cls
    segments.append(Segment(group, group_cls))
    return segments
