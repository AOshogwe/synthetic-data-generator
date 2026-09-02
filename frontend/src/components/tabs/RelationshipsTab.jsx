import { useAppDispatch, useAppState } from '../../state/AppContext'

export default function RelationshipsTab() {
  const { config } = useAppState()
  const dispatch = useAppDispatch()

  function setField(field, value) {
    dispatch({ type: 'SET_CONFIG_FIELD', field, value })
  }

  return (
    <div className="space-y-8">
      <div className="config-panel rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">
          <i className="fas fa-clock mr-2"></i>Temporal Relationships
        </h3>
        <label className="flex items-center mb-4">
          <input
            type="checkbox"
            className="mr-3"
            checked={config.preserve_temporal}
            onChange={(e) => setField('preserve_temporal', e.target.checked)}
          />
          <span>Preserve temporal relationships between date columns</span>
        </label>
        <p className="text-sm text-gray-600">
          Automatically detects and maintains relationships like admission → discharge dates
        </p>
      </div>

      <div className="config-panel rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">
          <i className="fas fa-project-diagram mr-2"></i>Conditional Dependencies
        </h3>
        <label className="flex items-center mb-4">
          <input
            type="checkbox"
            className="mr-3"
            checked={config.preserve_dependencies}
            onChange={(e) => setField('preserve_dependencies', e.target.checked)}
          />
          <span>Preserve conditional dependencies</span>
        </label>
        <p className="text-sm text-gray-600">
          Maintains relationships where one column's value affects another (e.g., test type affecting duration)
        </p>
      </div>

      <div className="config-panel rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">
          <i className="fas fa-chart-line mr-2"></i>Correlation Preservation
        </h3>
        <label className="flex items-center mb-4">
          <input
            type="checkbox"
            className="mr-3"
            checked={config.preserve_correlations}
            onChange={(e) => setField('preserve_correlations', e.target.checked)}
          />
          <span>Preserve statistical correlations</span>
        </label>
      </div>
    </div>
  )
}
