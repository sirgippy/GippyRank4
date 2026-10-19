// Legacy Performance links and metadata select the unchanged Quality of Play.
export function variantFor(entry) {
  return entry.performance_variant ?? "quality_of_play";
}

export function requestedVariant(params) {
  if (params.has("variant")) return params.get("variant") === "quality_of_record" ? "quality_of_record" : "quality_of_play";
  return params.get("snapshot")?.endsWith("-performance-quality-of-record") ? "quality_of_record" : "quality_of_play";
}

export function performanceLabel(entry) {
  return variantFor(entry) === "quality_of_record" ? "Quality of Record" : "Quality of Play";
}
