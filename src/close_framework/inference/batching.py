"""Batching helpers for scalable report processing."""

from __future__ import annotations

from itertools import islice
from time import perf_counter
from typing import Iterable, Iterator, List, Sequence, TypeVar

T = TypeVar("T")


def batched(items: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    """Yield lists of at most ``batch_size`` items."""

    iterator = iter(items)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            return
        yield batch


class Timer:
    """Small wall-clock timer for benchmarking stages."""

    def __enter__(self) -> "Timer":
        self.start = perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed = perf_counter() - self.start


def flatten(batches: Sequence[Sequence[T]]) -> List[T]:
    """Flatten a list of batches."""

    return [item for batch in batches for item in batch]
