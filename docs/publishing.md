# Publishing this source release

This source tree combines the UniARC and XARES-LLM implementations in one
repository. Integration into the existing UniARC repository uses a branch and
pull request, retaining its original commit history. Review the new directory
layout and removal of generated historical artifacts before merging.

After extracting the source archive into a new directory, these are the local
initialization commands:

```bash
git init -b main
git add .
git diff --cached --stat
git commit -m "Integrate UniARC probing and XARES-LLM evaluation"
```

The provided working directory may already be initialized and staged. Use your
normal Git identity for the commit. After creating an empty GitHub repository,
set its actual URL as `origin` and push `main`. Do not force-push an existing
repository to install this snapshot.

Suggested description:

> Unified comparison of discrete and continuous audio representations with
> XARES-LLM LoRA evaluation and frozen-backbone probing.

The source release can be shared as code with its documented validation limits.
For a paper reproduction release, add the verified paper citation, original
model revisions, dataset split definitions, missing K-means centroids/recipe,
and real-GPU validation results. Share external assets only under their own
terms. Keep the limitations in the README visible until those artifacts and
results are actually available.
