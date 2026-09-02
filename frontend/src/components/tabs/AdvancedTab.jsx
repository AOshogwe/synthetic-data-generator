import { useAppDispatch, useAppState } from '../../state/AppContext'
import { PRIVACY_LEVELS } from '../../config/contract'

export default function AdvancedTab() {
  const { config } = useAppState()
  const dispatch = useAppDispatch()

  function setField(field, value) {
    dispatch({ type: 'SET_CONFIG_FIELD', field, value })
  }

  return (
    <div className="grid md:grid-cols-2 gap-8">
      <div className="config-panel rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">
          <i className="fas fa-lock mr-2"></i>Privacy Settings
        </h3>
        <div className="space-y-3">
          <label>Privacy level:</label>
          <select
            className="w-full p-3 border border-gray-300 rounded-lg"
            value={config.privacy_level}
            onChange={(e) => setField('privacy_level', e.target.value)}
          >
            {PRIVACY_LEVELS.map((level) => (
              <option key={level.value} value={level.value}>
                {level.label} - {level.description}
              </option>
            ))}
          </select>
          <label className="flex items-center mt-3">
            <input
              type="checkbox"
              className="mr-3"
              checked={config.differential_privacy}
              onChange={(e) => setField('differential_privacy', e.target.checked)}
            />
            <span>Apply differential privacy</span>
          </label>
          <p className="text-xs text-gray-500">
            Adds Laplace-mechanism noise (epsilon=1.0) to numeric columns. This is a real, working noise
            mechanism, but not a formally audited differential-privacy guarantee -- treat it as a strong
            additional safeguard, not a compliance certification.
          </p>
        </div>
      </div>

      <div className="config-panel rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">
          <i className="fas fa-shield-alt mr-2"></i>Data Quality
        </h3>
        <div className="space-y-3">
          <label className="flex items-center">
            <input
              type="checkbox"
              className="mr-3"
              checked={config.handle_missing_values}
              onChange={(e) => setField('handle_missing_values', e.target.checked)}
            />
            <span>Handle missing values</span>
          </label>
          <label className="flex items-center">
            <input
              type="checkbox"
              className="mr-3"
              checked={config.remove_duplicate_rows}
              onChange={(e) => setField('remove_duplicate_rows', e.target.checked)}
            />
            <span>Remove duplicate rows</span>
          </label>
          <label className="flex items-center">
            <input
              type="checkbox"
              className="mr-3"
              checked={config.detect_and_handle_outliers}
              onChange={(e) => setField('detect_and_handle_outliers', e.target.checked)}
            />
            <span>Detect and handle outliers</span>
          </label>
        </div>
      </div>
    </div>
  )
}
