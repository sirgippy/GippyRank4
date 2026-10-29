// Legacy Performance links and metadata select the unchanged Quality of Play.
export function variantFor(entry) {
  return entry.performance_variant ?? "quality_of_play";
}

export function requestedVariant(params) {
  const variant = params.get("variant");
  if (variant) return ["quality_of_record", "quality_of_results"].includes(variant) ? variant : "quality_of_play";
  const snapshot = params.get("snapshot");
  if (snapshot?.endsWith("-performance-quality-of-results")) return "quality_of_results";
  return snapshot?.endsWith("-performance-quality-of-record") ? "quality_of_record" : "quality_of_play";
}

export function performanceLabel(entry) {
  return { quality_of_play: "Quality of Play", quality_of_record: "Quality of Record", quality_of_results: "Quality of Results" }[variantFor(entry)] ?? "Quality of Play";
}
