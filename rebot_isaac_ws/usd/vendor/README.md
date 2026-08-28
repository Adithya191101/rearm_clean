# Seeed Isaac Sim Asset

`scripts/fetch_sources.sh` sparsely checks out
[`Seeed-Projects/reBot-Isaacsim`](https://github.com/Seeed-Projects/reBot-Isaacsim)
at the revision in `dependencies.lock`, then creates:

```text
rebot_isaac_ws/usd/vendor/reBot_B601_DM
  -> ../../.upstream/reBot-Isaacsim/usd/reBot_B601_DM
```

The generated directory is ignored by this repository. The runtime verifies
the official root layer SHA-256:

```text
6b9d39de1200732c581c91e895bee412844e101006fb0c3df54259d81ee28e84
```

The asset remains under Seeed Studio's upstream terms.
