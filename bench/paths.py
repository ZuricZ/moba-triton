"""Where a measurement belongs, decided by the card that produced it.

WHY THIS MODULE EXISTS. `results/` was one flat directory, and the card a
number came from lived in the filename when it lived anywhere at all --
`block-sweep-budget-3050ti.csv` said so, `moba-fwd.csv` did not, and the
second is the file `docs/why-we-beat-flashmoba.md` cites for its headline
table. That is not merely untidy. `triton.testing.perf_report(save_path=...)`
writes a *fixed* set of names into whatever directory it is handed, so running
`bench.compare` on a second card overwrites the first card's CSV in place,
with no error and no name collision to notice. An A100 claim would then be
standing on H100 data.

Making the card a directory rather than a suffix fixes both at once: a path
names its own provenance, and two cards cannot collide however fixed the
filenames are.

LAYOUT.

    results/<card>/data/        csv, json -- the measurements
    results/<card>/figures/     png -- drawn from that card's data
    results/<card>/logs/        the raw session transcripts

DETECTION AND OVERRIDE. The writing side asks the driver which card it is on,
because a benchmark always knows. The *reading* side must not: figures for the
A100 are drawn on a laptop that has never seen one, so every plotting script
takes `--card` and only falls back to the local device. A reader that guessed
from the local GPU would silently draw the wrong card's data, which is the
failure this module exists to prevent.
"""

import argparse
import os

# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------

# Architecture tag -> directory name. The board is in the name for a human
# reading a path; the architecture is what actually selects tuned configs, so
# it leads. Only cards this project has run on appear -- an unknown card gets
# its bare tag rather than a guessed board name.
CARD_DIRS = {
    "sm_80": "sm80-a100",
    "sm_86": "sm86-3050ti",
    "sm_90": "sm90-h100",
}

ROOT = "results"

KINDS = ("data", "figures", "logs")


def local_card() -> str:
    """Returns the directory name for the GPU in this machine.

    Returns:
        A member of `CARD_DIRS`' values, or the bare tag ("sm89") for a card
        this project has not run on, or "unknown" with no CUDA device. Never
        raises: a path helper is not the place to discover there is no GPU,
        and the plotting scripts must import on a machine without one.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return "unknown"
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return "unknown"
    tag = f"sm_{major}{minor}"
    return CARD_DIRS.get(tag, tag.replace("_", ""))


def card_dir(card: str | None = None, kind: str = "data",
             create: bool = False) -> str:
    """Returns the directory one kind of output belongs in.

    Args:
        card: Directory name such as "sm80-a100"; the local card when None.
        kind: One of "data", "figures", "logs".
        create: Whether to make the directory. True on the writing side,
            False on the reading side -- a reader that creates the directory
            it failed to find turns "no such measurement" into an empty
            listing, which is a worse error message.

    Returns:
        A path like "results/sm80-a100/data".

    Raises:
        ValueError: If `kind` is not one of `KINDS`.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    path = os.path.join(ROOT, card or local_card(), kind)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def device(card: str | None = None) -> dict:
    """Returns what `nvidia-smi` recorded about the card that made the results.

    WHY THE READING SIDE NEEDS THIS. A figure caption naming a card is a claim
    about provenance, and until now those claims were module-level string
    constants: `bench/sweep_plot.py` said "on an A100-40GB" on every figure it
    drew, including the H100's, where it was simply false. The card that
    produced a directory of results is recorded in that directory, by the
    session that produced it -- so a caption can read it instead of asserting
    it, and cannot drift out of date the way a constant did.

    The file is the header-and-one-row CSV `nvidia-smi --query-gpu` writes:

        name, memory.total [MiB], compute_cap, driver_version
        NVIDIA H100 80GB HBM3, 81559 MiB, 9.0, 550.163.01

    Args:
        card: Directory name such as "sm80-a100"; the local card when None.

    Returns:
        A dict with "name" ("NVIDIA H100 80GB HBM3"), "short" (the same with
        the vendor stripped, which is how this project's prose writes it),
        "memory_mib" and "memory_gib" (int, rounded), "compute_cap" and
        "driver". Empty when the card kept no `device.txt` -- sm86-3050ti did
        not, and an empty dict is a caller's cue to leave the clause out
        rather than guess it.
    """
    path = os.path.join(card_dir(card, "logs"), "device.txt")
    try:
        with open(path, encoding="utf-8") as handle:
            rows = [line.strip() for line in handle if line.strip()]
    except OSError:
        return {}
    if len(rows) < 2:
        return {}

    fields = [field.strip() for field in rows[1].split(",")]
    if not fields or not fields[0]:
        return {}
    facts = {"name": fields[0], "short": fields[0].removeprefix("NVIDIA ")}
    if len(fields) > 1 and fields[1].split()[:1]:
        try:
            mib = int(fields[1].split()[0])
        except ValueError:
            mib = 0
        if mib:
            facts["memory_mib"] = mib
            facts["memory_gib"] = round(mib / 1024)
    if len(fields) > 2:
        facts["compute_cap"] = fields[2]
    if len(fields) > 3:
        facts["driver"] = fields[3]
    return facts


def known_cards(kind: str = "data") -> list[str]:
    """Returns the cards that actually have results on disk.

    Used to turn "file not found" into a message naming what *is* available,
    which is the difference between a dead end and a usable one.

    Args:
        kind: Which subdirectory must be non-empty to count.

    Returns:
        Directory names, sorted.
    """
    if not os.path.isdir(ROOT):
        return []
    return sorted(
        name for name in os.listdir(ROOT)
        if os.path.isdir(os.path.join(ROOT, name, kind))
        and os.listdir(os.path.join(ROOT, name, kind))
    )


def add_card_argument(parser: argparse.ArgumentParser) -> None:
    """Adds the `--card` flag, with the same help text everywhere.

    Args:
        parser: The parser to extend.
    """
    parser.add_argument(
        "--card", default=None,
        help="which card's results to read, e.g. sm80-a100. Defaults to the "
             "GPU in this machine, which is wrong whenever you are drawing "
             "a rented box's data on a laptop -- pass it explicitly then. "
             f"Present on disk: {', '.join(known_cards()) or 'none'}")
