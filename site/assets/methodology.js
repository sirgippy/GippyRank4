const displayFeatureStatus = (value) => {
  if (typeof value !== "string") {
    return value;
  }
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
};

const loadMethodologyMetadata = async () => {
  try {
    const response = await fetch("./data/methodology.json");
    if (!response.ok) {
      throw new Error(`Metadata request failed: ${response.status}`);
    }
    const metadata = await response.json();

    document.querySelectorAll("[data-version-key]").forEach((element) => {
      const value = metadata.model_versions?.[element.dataset.versionKey];
      if (typeof value === "string") {
        element.textContent = `${element.dataset.versionPrefix ?? ""}${value}`;
      }
    });

    document.querySelectorAll("[data-schema-key]").forEach((element) => {
      const value = metadata.schema_versions?.[element.dataset.schemaKey];
      if (typeof value === "string") {
        element.textContent = `${element.dataset.schemaPrefix ?? ""}${value}`;
      }
    });

    document.querySelectorAll("[data-feature-key]").forEach((element) => {
      const value = metadata.feature_status?.[element.dataset.featureKey];
      if (typeof value === "string") {
        element.textContent = displayFeatureStatus(value);
      }
    });
  } catch (error) {
    // The HTML contains readable fallbacks, so a missing data file does not
    // make the methodology page blank or misleading.
    console.warn("Methodology metadata unavailable", error);
  }
};

loadMethodologyMetadata();
