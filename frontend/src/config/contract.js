// Single source of truth for the /api/configure payload shape.
//
// This exists because of a bug pattern that hit this codebase at least six
// times (tasks #17, #18, #22, #30, #31, #33): a control would render in the
// UI but nothing kept it connected to what the backend actually reads, and
// nothing caught the gap until someone manually traced that one setting end
// to end. CONFIG_FIELDS below is the complete, exhaustive list of keys this
// UI is allowed to send. buildConfigPayload() can only ever emit exactly
// this shape, and app.py's /api/configure logs a warning (see
// validate_config_payload in app.py) if it ever receives a key not in the
// matching Python-side list -- so a new field added to only one side is
// caught immediately instead of silently doing nothing.
//
// Deliberately NOT included: privacy_level's differential_privacy epsilon
// override (fixed at 1.0 server-side for now). handle_missing_values,
// remove_duplicate_rows, detect_and_handle_outliers, and correlation_level
// were found decorative during this rebuild (task #46) and now have a real
// backend implementation (pipeline.py's apply_data_quality_options and
// apply_correlation_preservation), so they're included below.

export const DEFAULT_CONFIG = {
  generation_method: 'auto',
  data_size: { type: 'same' },
  anonymize_names: false,
  name_method: 'synthetic',
  preserve_gender: false,
  apply_age_grouping: false,
  age_grouping_method: '10-year',
  anonymize_addresses: false,
  address_method: 'remove_house_number',
  perturbation_factor: 0.2,
  preserve_temporal: true,
  preserve_dependencies: true,
  preserve_correlations: true,
  correlation_level: 'moderate',
  privacy_level: 'balanced',
  differential_privacy: false,
  handle_missing_values: true,
  remove_duplicate_rows: false,
  detect_and_handle_outliers: false,
  // { [tableName]: { [columnName]: 'synthesize' | 'copy' | 'range' | 'abstract' } }
  column_selection: {},
}

export const GENERATION_METHODS = [
  { value: 'auto', label: 'Auto', description: 'Intelligent method selection' },
  { value: 'perturbation', label: 'Perturbation', description: 'Controlled modifications (fastest)' },
  { value: 'ctgan', label: 'CTGAN', description: 'Deep learning-based generation' },
  { value: 'gaussian_copula', label: 'Gaussian Copula', description: 'Statistical modeling' },
]

export const DATA_SIZE_TYPES = [
  { value: 'same', label: 'Same as original' },
  { value: 'percentage', label: 'Percentage of original' },
  { value: 'custom', label: 'Custom number of rows' },
]

export const NAME_METHODS = [
  { value: 'synthetic', label: 'Synthetic names' },
  { value: 'initials', label: 'Initials only' },
  { value: 'random', label: 'Random shuffle' },
]

export const AGE_GROUPING_METHODS = [
  { value: '5-year', label: '5-year bands' },
  { value: '10-year', label: '10-year bands' },
  { value: 'life-stages', label: 'Life stages' },
]

// Matches models/address_synthesis.py's AddressSynthesizer.anonymize_address
// method names exactly -- 'city_only' here previously didn't match any real
// method name, so picking it silently fell through to "leave value
// unchanged" (task #35).
export const ADDRESS_METHODS = [
  { value: 'remove_house_number', label: 'Remove house number', description: 'Keep street, city, state, zip' },
  { value: 'street_only', label: 'Street name only', description: 'Drop house number and everything after the street' },
  { value: 'city_state_only', label: 'City and state only', description: 'Drop street and zip' },
  { value: 'zip_only', label: 'Zip code only', description: 'Drop everything but the zip code' },
  { value: 'general_area', label: 'General area', description: 'Keep city, state, and zip' },
  { value: 'synthesize_realistic', label: 'Synthesize realistic address', description: 'Generate a plausible fake address' },
]

export const CORRELATION_LEVELS = [
  { value: 'strict', label: 'Strict', description: 'Maintain exact correlations' },
  { value: 'moderate', label: 'Moderate', description: 'Allow some variation' },
  { value: 'loose', label: 'Loose', description: 'Preserve general trends only' },
]

export const PRIVACY_LEVELS = [
  { value: 'minimal', label: 'Minimal', description: 'Preserve data utility' },
  { value: 'balanced', label: 'Balanced', description: 'Balance privacy and utility' },
  { value: 'high', label: 'High', description: 'Maximum privacy protection (forces name/age/address anonymization)' },
]

export const COLUMN_ACTIONS = [
  { value: 'synthesize', label: 'Synthesize' },
  { value: 'copy', label: 'Copy original' },
  { value: 'range', label: 'Convert to range' },
  { value: 'abstract', label: 'Abstract/anonymize' },
]

/**
 * Build the exact /api/configure request body from app state. This is the
 * ONLY place that payload gets constructed -- unlike the old
 * collectConfiguration(), which read up to 20 scattered DOM elements by id
 * at submit time, this reads one state object whose shape is defined above.
 */
export function buildConfigPayload(config) {
  const payload = {
    generation_method: config.generation_method,
    data_size: { ...config.data_size },
    anonymize_names: config.anonymize_names,
    apply_age_grouping: config.apply_age_grouping,
    anonymize_addresses: config.anonymize_addresses,
    perturbation_factor: config.perturbation_factor,
    preserve_temporal: config.preserve_temporal,
    preserve_dependencies: config.preserve_dependencies,
    preserve_correlations: config.preserve_correlations,
    correlation_level: config.correlation_level,
    privacy_level: config.privacy_level,
    differential_privacy: config.differential_privacy,
    handle_missing_values: config.handle_missing_values,
    remove_duplicate_rows: config.remove_duplicate_rows,
    detect_and_handle_outliers: config.detect_and_handle_outliers,
    column_selection: config.column_selection,
  }

  if (config.apply_age_grouping) {
    payload.age_grouping_method = config.age_grouping_method
  }
  if (config.anonymize_names) {
    payload.name_method = config.name_method
    payload.preserve_gender = config.preserve_gender
  }
  if (config.anonymize_addresses) {
    payload.address_method = config.address_method
  }

  return payload
}
