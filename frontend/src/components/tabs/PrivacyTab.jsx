import { useAppDispatch, useAppState } from '../../state/AppContext'
import { NAME_METHODS, AGE_GROUPING_METHODS, ADDRESS_METHODS } from '../../config/contract'

export default function PrivacyTab() {
  const { config } = useAppState()
  const dispatch = useAppDispatch()

  function setField(field, value) {
    dispatch({ type: 'SET_CONFIG_FIELD', field, value })
  }

  return (
    <div>
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-6 text-sm text-blue-700">
        A column explicitly set to "Copy original" in Column Selection is never touched by these toggles,
        even if it matches a name/age/address pattern below.
      </div>

      <div className="grid md:grid-cols-2 gap-8">
        <div className="config-panel rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">👤 Name Anonymization</h3>
          <label className="flex items-center mb-3">
            <input
              type="checkbox"
              className="mr-3"
              checked={config.anonymize_names}
              onChange={(e) => setField('anonymize_names', e.target.checked)}
            />
            <span>Anonymize name columns</span>
          </label>
          {config.anonymize_names && (
            <div className="space-y-3 ml-6">
              <select
                className="w-full p-3 border border-gray-300 rounded-lg"
                value={config.name_method}
                onChange={(e) => setField('name_method', e.target.value)}
              >
                {NAME_METHODS.map((m) => (
                  <option key={m.value} value={m.value}>
                    {m.label}
                  </option>
                ))}
              </select>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  className="mr-3"
                  checked={config.preserve_gender}
                  onChange={(e) => setField('preserve_gender', e.target.checked)}
                />
                <span>Preserve gender association</span>
              </label>
            </div>
          )}
        </div>

        <div className="config-panel rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">🎂 Age Grouping</h3>
          <label className="flex items-center mb-3">
            <input
              type="checkbox"
              className="mr-3"
              checked={config.apply_age_grouping}
              onChange={(e) => setField('apply_age_grouping', e.target.checked)}
            />
            <span>Group ages into ranges</span>
          </label>
          {config.apply_age_grouping && (
            <select
              className="w-full p-3 border border-gray-300 rounded-lg ml-6"
              style={{ width: 'calc(100% - 1.5rem)' }}
              value={config.age_grouping_method}
              onChange={(e) => setField('age_grouping_method', e.target.value)}
            >
              {AGE_GROUPING_METHODS.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
          )}
          <p className="text-xs text-gray-500 mt-2">
            Any Date-of-Birth column is automatically degraded to a matching birth-year range when this is on.
          </p>
        </div>

        <div className="config-panel rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">📍 Address Anonymization</h3>
          <label className="flex items-center mb-3">
            <input
              type="checkbox"
              className="mr-3"
              checked={config.anonymize_addresses}
              onChange={(e) => setField('anonymize_addresses', e.target.checked)}
            />
            <span>Anonymize address/postal columns</span>
          </label>
          {config.anonymize_addresses && (
            <select
              className="w-full p-3 border border-gray-300 rounded-lg ml-6"
              style={{ width: 'calc(100% - 1.5rem)' }}
              value={config.address_method}
              onChange={(e) => setField('address_method', e.target.value)}
            >
              {ADDRESS_METHODS.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
          )}
        </div>

        <div className="config-panel rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">🎚️ Perturbation Level</h3>
          <input
            type="range"
            min="0"
            max="100"
            value={Math.round(config.perturbation_factor * 100)}
            onChange={(e) => setField('perturbation_factor', parseFloat(e.target.value) / 100)}
            className="w-full"
          />
          <p className="text-center text-sm text-gray-600 mt-2">{Math.round(config.perturbation_factor * 100)}%</p>
        </div>
      </div>
    </div>
  )
}
