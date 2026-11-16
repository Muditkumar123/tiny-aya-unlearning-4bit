# Release history and the merged layout

The current project uses folders by purpose: `training/`, `compression/`, `evaluation/`, `shared/`, `configs/`, `data/`, `results/` and `docs/`. The earlier `Mudit/`, `Ganesh/`, `Vedika/` and `common/` folders are no longer required to run this layout.

The complete earlier release remains available at commit [`0e4078e`](https://github.com/Muditkumar123/tiny-aya-unlearning-4bit/tree/0e4078e604b366a51dbee047aea244e1bf4c6530). Removing the duplicate folders from the current branch does not remove their commit history or the students' original commits.

| Earlier folder | Current location |
| --- | --- |
| `Ganesh/scripts/`, `Ganesh/configs/` | `training/`, `configs/` |
| `Mudit/scripts/` | `compression/` |
| `Vedika/scripts/` | `evaluation/` |
| `common/addons/` | `shared/` |
| `Vedika/release/export/` | `results/` |
| Training and compression measurement records | `results/provenance/training/`, `results/provenance/compression/` |

The [file-by-file map](history/layout_map.csv) lists all 166 removed paths, hashes of their archived contents, current counterparts where available, and permanent links to the original files. Counterparts labelled "adapted" have changed paths, formatting or presentation; the archived file is the exact historical source. Historical launchers, superseded builders, release-planning notes and owner write-ups remain in that archive rather than the current runnable layout.

## Historical annotation provenance

These three notes are retained byte for byte in the current release because they explain the limits of the exploratory annotation comparison:

- [Frozen exploratory plan](history/j3/J3_EXPLORATORY_PLAN_20261114.public.md)
- [Exploratory results and limitations](history/j3/J3_EXPLORATORY_RESULTS_20261114.public.md)
- [Later self-report addendum](history/j3/J3_PUBLIC_ADDENDUM_20261114.md)

Paths inside those historical notes refer to the archived layout. They do not establish independent human validation of the automatic judge.

For current execution instructions, use [Reproducing the project](REPRODUCING.md). For licences and attribution, see the root [LICENSE](../LICENSE), [source notices](../data/SOURCES.md), and the model cards. This layout cleanup changes no model weights, numerical results, training settings or scientific conclusions.
