# Integration and provenance

## Source snapshots

| Original source | Commit | Destination |
| --- | --- | --- |
| [UniARC](https://github.com/Anachronism-N/UniARC/tree/46ab7b41907c3e85c2fbd53b9ac225e27ce9902e/code) `code/` | `46ab7b41907c3e85c2fbd53b9ac225e27ce9902e` | `uniarc/` |
| [UniARC_paper](https://github.com/Anachronism-N/UniARC_paper/tree/49f21313d030ff573e50631f4a4ba065e5729d79/xares-llm) `xares-llm/` | `49f21313d030ff573e50631f4a4ba065e5729d79` | `xares-llm/` |

This is a curated source integration based on the existing UniARC repository.
The integration branch retains UniARC's prior commits and imports the selected
XARES-LLM source files at the pinned UniARC_paper commit. It does not import the
second repository's full Git history or retain every historical artifact in the
new source tree. Both original commits remain the provenance record.

## Organization

The paper calls the combined framework **UniARC**. The two backend directories
represent its two evaluation strategies, not two competing paper titles.
One root launcher routes to either backend in a separate process, preserving its
imports and environment. The two dependency stacks are intentionally installed
in separate virtual environments.

Backend READMEs describe portability fixes. The integration does not supply new
experimental measurements. Dry runs and static tests are not paper reproduction.

## Excluded material

Training logs, TensorBoard events, old predictions, debug scripts, Python caches,
editor files, bundled Git backups, downloaded model directories, datasets,
paper LaTeX submission assets, and unrelated encoder experiments are excluded.
In particular, case-colliding historical `speechtokenizer_US`/`speechtokenizer_us`
output directories in the old UniARC tree are not source code and are omitted.
External encoder implementations and checkpoints must be obtained separately.

The full local manuscript is not redistributed with the code. Its current title
is **Discrete vs. Continuous: A Comprehensive Study of Unified Audio Understanding
in LALMs**. The older title and publication citation in the source repository
README were not treated as authoritative. `CITATION.cff` contains a software
citation until final paper metadata is confirmed.

## License

The project owners selected Apache-2.0 for their original code and integration
changes. XARES-LLM's Apache-2.0 license and original headers are preserved.
Downloaded datasets and checkpoints are separate assets with their own terms.
See [NOTICE](../NOTICE) and [third-party attribution](third_party.md).
