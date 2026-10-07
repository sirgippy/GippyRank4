# Shared research-data cache

Research scripts that retain large raw source corpora use one data root outside
the repository, so separate worktrees can reuse the same downloads.

The root is configured by `GIPPYRANK_DATA_DIR`. When unset, it defaults to
`$XDG_CACHE_HOME/gippyrank/research-data`, or to
`~/.cache/gippyrank/research-data` when `XDG_CACHE_HOME` is unset. The issue 183
CFBD corpus is stored under:

```text
<research-data-root>/raw/cfbd/offensive_line_shared_roster_issue_183/
```

Both the acquisition and offline builder scripts accept `--raw-root` to use a
different corpus directory for a single run. Raw responses are immutable after
acquisition: reruns verify and reuse cached bytes, and only uncached requests
need `CFBD_API_KEY`. The acquisition manifest and the builder's source inventory
store response paths relative to the raw root (`path_base=raw_root`), not to a
particular repository checkout.

Normal CI is independent of this external cache. It does not acquire or rebuild
the CFBD corpus; acquisition and research-output rebuilds are explicit local
operations.
