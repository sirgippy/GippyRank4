# Issue 212: replay incomplete

No candidate scores were interpreted because a frozen input, inference, or parity check failed. This is not a negative or inconclusive scientific result. Restore the required artifact through the documented workflow, verify its hash, and rerun the fixed protocol.

Blocking reason: `#195 input provenance verification failed: issue200_manifest: expected sha256=09dd40ec9e5164b923e1970fa8d3d99ffe27386d279519c79c48b16da04f6047 bytes=49238, got sha256=12bb43aa49009a849b0ca3ebdf8110f92fa6c3bff6df1bcc32aa1e473c08dbe5 bytes=49089; issue200_preseason_feature_manifest: expected sha256=0f1e327d293c441427a791d258527a5220afe03524b8a5400cfe1262251e78dd bytes=381252, got sha256=151154319618ba6b61eb78738464e1ab76800db0bb4300d0d5d21a065d73cfd4 bytes=326784; issue200_context_feature_inputs: missing at data/research/matched_history_context_issue_200/context_feature_inputs.csv; #195 recorded #200 manifest hash does not match`
