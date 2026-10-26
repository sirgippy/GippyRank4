import { performanceLabel, variantFor } from "./ranking-variants.js";

const families = new Set(["predictive", "performance"]);
const priors = new Set(["context", "history"]);
const variants = new Set(["quality_of_play", "quality_of_record"]);

function entryLabel(entry) {
  const family = entry.ranking_family === "performance"
    ? performanceLabel(entry)
    : `Predictive ${entry.prior_family === "history" ? "History" : "Context"}`;
  return `${entry.season} ${entry.display_label} · ${family}`;
}

function result(selection, entry = null, error = null, alternatives = []) {
  return { ...selection, entry, error, alternatives: alternatives.slice(0, 4) };
}

function candidatesFor(manifest, selection, slot = null) {
  return manifest.snapshots.filter((entry) => (
    entry.season === selection.season
    && entry.ranking_family === selection.family
    && (selection.family !== "predictive" || entry.prior_family === selection.prior)
    && (selection.family !== "performance" || variantFor(entry) === selection.variant)
    && (slot === null || entry.publication_slot === slot)
  ));
}

function matchingAlternative(manifest, selection) {
  const entries = candidatesFor(manifest, selection);
  return entries.find((entry) => entry.publication_slot === manifest.default_publication_slot) ?? entries[0] ?? null;
}

function selectorAlternatives(manifest, selection) {
  const slotEntries = selection.requested === null
    ? []
    : manifest.snapshots.filter((entry) => entry.season === selection.season && entry.publication_slot === selection.requested);
  if (slotEntries.length) return slotEntries;
  const alternative = matchingAlternative(manifest, selection);
  return alternative ? [alternative] : [];
}

function mismatchResult(selection, entry, mismatches) {
  return result(selection, null, {
    code: "selector-mismatch",
    message: mismatches.join(" "),
  }, [entry]);
}

export function resolvePublication(manifest, params) {
  const hasSnapshot = params.has("snapshot");
  const hasSlot = params.has("slot");
  const raw = hasSnapshot ? params.get("snapshot") : hasSlot ? params.get("slot") : null;
  const exact = raw === null ? null : manifest.snapshots.find((entry) => entry.snapshot_id === raw) ?? null;
  const slotEntries = raw === null ? [] : manifest.snapshots.filter((entry) => entry.publication_slot === raw);
  const slotSeasons = [...new Set(slotEntries.map((entry) => entry.season))];
  const firstSeason = manifest.seasons[0];
  const familyValue = params.get("family");
  const familyExplicit = params.has("family");
  const seasonExplicit = params.has("season");
  const priorExplicit = params.has("prior");
  const variantExplicit = params.has("variant");

  let family = exact && !familyExplicit
    ? exact.ranking_family
    : familyExplicit
      ? familyValue
      : variantExplicit
        ? "performance"
        : "predictive";
  let season = exact && !seasonExplicit
    ? exact.season
    : seasonExplicit
      ? Number(params.get("season"))
      : slotSeasons.length === 1
        ? slotSeasons[0]
        : firstSeason;
  let prior = exact && !priorExplicit && exact.ranking_family === "predictive"
    ? exact.prior_family
    : params.get("prior") ?? "context";
  let variant = exact && !variantExplicit && exact.ranking_family === "performance"
    ? variantFor(exact)
    : params.get("variant") ?? (raw?.endsWith("-performance-quality-of-record") ? "quality_of_record" : "quality_of_play");
  const selection = { family, season, prior, variant, requested: raw, explicit: hasSnapshot || hasSlot };

  if (hasSnapshot && hasSlot && params.get("snapshot") !== params.get("slot")) {
    return result(selection, null, {
      code: "conflicting-selectors",
      message: "The snapshot and slot parameters request different publications.",
    });
  }
  if (familyExplicit && !families.has(family)) {
    return result(selection, null, { code: "invalid-family", message: `The requested ranking family “${family}” is unavailable.` });
  }
  if (seasonExplicit && (!Number.isSafeInteger(season) || !manifest.seasons.includes(season))) {
    return result(selection, null, { code: "invalid-season", message: `The requested season “${params.get("season")}" is unavailable.` });
  }
  if (priorExplicit && !priors.has(prior)) {
    return result(selection, null, { code: "invalid-prior", message: `The requested Predictive starting point “${prior}” is unavailable.` });
  }
  if (variantExplicit && !variants.has(variant)) {
    return result(selection, null, { code: "invalid-variant", message: `The requested Performance variant “${variant}” is unavailable.` });
  }

  if (exact) {
    const mismatches = [];
    if (familyExplicit && family !== exact.ranking_family) {
      mismatches.push(`Snapshot “${raw}” belongs to ${exact.ranking_family === "performance" ? "Performance" : "Predictive"}, but ${family === "performance" ? "Performance" : "Predictive"} was requested.`);
    }
    if (seasonExplicit && season !== exact.season) {
      mismatches.push(`Snapshot “${raw}” belongs to season ${exact.season}, but season ${season} was requested.`);
    }
    if (priorExplicit && (exact.ranking_family !== "predictive" || prior !== exact.prior_family)) {
      mismatches.push(`Snapshot “${raw}” uses ${exact.ranking_family === "predictive" ? `${exact.prior_family} starting point` : "Performance, which has no Predictive starting point"}, but ${exact.ranking_family === "predictive" ? `${prior} was requested` : `prior=${prior} was requested`}.`);
    }
    if (variantExplicit && (exact.ranking_family !== "performance" || variant !== variantFor(exact))) {
      mismatches.push(`Snapshot “${raw}” is ${exact.ranking_family === "performance" ? performanceLabel(exact) : "Predictive"}, but ${performanceLabel({ performance_variant: variant })} was requested.`);
    }
    if (mismatches.length) return mismatchResult(selection, exact, mismatches);
    selection.family = exact.ranking_family;
    selection.season = exact.season;
    selection.prior = exact.prior_family ?? "context";
    selection.variant = variantFor(exact);
    return result(selection, exact);
  }

  if (priorExplicit && family === "performance") {
    return result(selection, null, {
      code: "selector-mismatch",
      message: `The prior=${prior} selector applies to Predictive publications, not Performance.`,
    }, selectorAlternatives(manifest, selection));
  }
  if (variantExplicit && family !== "performance") {
    return result(selection, null, {
      code: "selector-mismatch",
      message: `The variant=${variant} selector applies to Performance publications, not Predictive.`,
    }, selectorAlternatives(manifest, selection));
  }

  if (selection.explicit) {
    if (!raw) {
      return result(selection, null, { code: "unknown-publication", message: "The requested snapshot or publication slot is empty." });
    }
    const slotEntries = manifest.snapshots.filter((entry) => entry.publication_slot === raw);
    const matches = candidatesFor(manifest, selection, raw);
    if (matches.length) return result(selection, matches[0]);

    if (!slotEntries.length) {
      const alternative = matchingAlternative(manifest, selection);
      const label = hasSnapshot ? "snapshot" : "publication slot";
      return result(selection, null, {
        code: "unknown-publication",
        message: `The requested ${label} “${raw}” is unavailable.`,
      }, alternative ? [alternative] : []);
    }

    const sameSeason = slotEntries.filter((entry) => entry.season === season);
    const sameFamily = sameSeason.filter((entry) => entry.ranking_family === family);
    const samePrior = sameFamily.filter((entry) => family !== "predictive" || entry.prior_family === prior);
    const alternatives = samePrior.length ? samePrior : sameFamily.length ? sameFamily : sameSeason.length ? sameSeason : slotEntries;

    if (family === "performance" && sameFamily.length && !sameFamily.some((entry) => variantFor(entry) === variant)) {
      const available = [...new Set(sameFamily.map(variantFor))];
      const labels = available.map((item) => performanceLabel({ performance_variant: item }));
      return result(selection, null, {
        code: "variant-unavailable",
        message: `${performanceLabel({ performance_variant: variant })} is unavailable for publication slot “${raw}”. Available variant${labels.length === 1 ? "" : "s"}: ${labels.join(", ")}.`,
      }, sameFamily);
    }
    if (family === "predictive" && sameFamily.length && !samePrior.length) {
      const available = [...new Set(sameFamily.map((entry) => entry.prior_family))];
      const labels = available.map((item) => item === "history" ? "History" : "Context");
      return result(selection, null, {
        code: "prior-unavailable",
        message: `Predictive ${prior === "history" ? "History" : "Context"} is unavailable for publication slot “${raw}”. Available starting point${labels.length === 1 ? "" : "s"}: ${labels.join(", ")}.`,
      }, sameFamily);
    }

    const actual = alternatives[0];
    const actualDescription = actual.ranking_family === "performance"
      ? performanceLabel(actual)
      : `Predictive ${actual.prior_family === "history" ? "History" : "Context"}`;
    const requestedDescription = family === "performance"
      ? performanceLabel({ performance_variant: variant })
      : `Predictive ${prior === "history" ? "History" : "Context"}`;
    let message = `The requested ${requestedDescription} publication is unavailable for slot “${raw}”.`;
    if (actual.season !== season) message = `Publication slot “${raw}” belongs to season ${actual.season}, not season ${season}.`;
    else if (actual.ranking_family !== family) message = `Publication slot “${raw}” has ${actualDescription}; ${requestedDescription} was requested.`;
    else if (family === "predictive" && actual.prior_family !== prior) message = `Publication slot “${raw}” has ${actualDescription}; ${requestedDescription} was requested.`;
    return result(selection, null, { code: "publication-mismatch", message }, alternatives);
  }

  const available = candidatesFor(manifest, selection);
  const preferred = available.find((entry) => entry.publication_slot === manifest.default_publication_slot);
  if (preferred || available[0]) return result(selection, preferred ?? available[0]);

  const broader = manifest.snapshots.filter((entry) => entry.season === season && entry.ranking_family === family);
  const alternatives = broader.length ? broader : manifest.snapshots.filter((entry) => entry.season === season);
  let message = `No ${family === "performance" ? performanceLabel({ performance_variant: variant }) : `Predictive ${prior === "history" ? "History" : "Context"}`} publication is available for season ${season}.`;
  if (family === "performance" && broader.length) {
    const availableVariants = [...new Set(broader.map(variantFor))].map((item) => performanceLabel({ performance_variant: item }));
    message = `${performanceLabel({ performance_variant: variant })} is unavailable for season ${season}. Available variant${availableVariants.length === 1 ? "" : "s"}: ${availableVariants.join(", ")}.`;
  } else if (family === "predictive" && broader.length) {
    const availablePriors = [...new Set(broader.map((entry) => entry.prior_family))].map((item) => item === "history" ? "History" : "Context");
    message = `Predictive ${prior === "history" ? "History" : "Context"} is unavailable for season ${season}. Available starting point${availablePriors.length === 1 ? "" : "s"}: ${availablePriors.join(", ")}.`;
  }
  return result(selection, null, { code: "publication-unavailable", message }, alternatives);
}

export function publicationParams(entry, params) {
  const next = new URLSearchParams(params);
  next.set("season", String(entry.season));
  next.set("snapshot", entry.snapshot_id);
  next.delete("slot");
  next.set("family", entry.ranking_family);
  if (entry.ranking_family === "predictive") {
    next.set("prior", entry.prior_family);
    next.delete("variant");
  } else {
    next.delete("prior");
    next.set("variant", variantFor(entry));
  }
  return next;
}

export function renderPublicationAlternatives(container, resolution, hrefFor) {
  container.replaceChildren();
  if (!resolution.error) return;
  container.append(document.createTextNode(resolution.error.message));
  if (!resolution.alternatives.length) return;
  const links = document.createElement("span");
  links.className = "publication-alternatives";
  links.append(document.createTextNode(" Available publications: "));
  resolution.alternatives.forEach((entry, index) => {
    if (index) links.append(document.createTextNode(" · "));
    const anchor = document.createElement("a");
    anchor.href = hrefFor(entry);
    anchor.textContent = entryLabel(entry);
    links.append(anchor);
  });
  container.append(links);
}
