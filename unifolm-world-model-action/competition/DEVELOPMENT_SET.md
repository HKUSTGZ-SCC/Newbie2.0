# Public development set

This directory contains five public development input caches and their
`manifest.json`. They are intentionally sufficient to run the supplied local
benchmark, but are not the final judge inputs.

Each cache stores the initial, already preprocessed image/state/action tensors
and the RNG state immediately before the first sampler call. Keep these files
unchanged. The launcher validates their schema, case name, frame stride,
configuration hash, resolution, video length and observation length.

The final judge mounts a private directory with the same layout after your
repository has been checked out. It does not use these public files.

The public package deliberately excludes raw PNG/H5/CSV source prompts,
reference videos and cache-generation utilities. They are not needed for
participant development or submission.
