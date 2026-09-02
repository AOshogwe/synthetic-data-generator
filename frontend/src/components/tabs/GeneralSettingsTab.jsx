import { useAppDispatch, useAppState } from '../../state/AppContext'
import { GENERATION_METHODS, DATA_SIZE_TYPES } from '../../config/contract'
import ColumnSelection from '../ColumnSelection'

export default function GeneralSettingsTab() {
  const { config } = useAppState()
  const dispatch = useAppDispatch()

  function setField(field, value) {
    dispatch({ type: 'SET_CONFIG_FIELD', field, value })
  }

  return (
    <div>
      <div className="grid md:grid-cols-2 gap-8">
        <div className="config-panel rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">🎯 Generation Method</h3>
          <div className="space-y-3">
            {GENERATION_METHODS.map((method) => (
              <label key={method.value} className="flex items-center">
                <input
                  type="radio"
                  name="generation-method"
                  className="mr-3"
                  checked={config.generation_method === method.value}
                  onChange={() => setField('generation_method', method.value)}
                />
                <span>
                  <strong>{method.label}</strong> - {method.description}
                </span>
              </label>
            ))}
          </div>
        </div>

        <div className="config-panel rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">📏 Output Data Size</h3>
          <div className="space-y-3">
            {DATA_SIZE_TYPES.map((opt) => (
              <label key={opt.value} className="flex items-center">
                <input
                  type="radio"
                  name="data-size"
                  className="mr-3"
                  checked={config.data_size.type === opt.value}
                  onChange={() => setField('data_size', { type: opt.value })}
                />
                <span>{opt.label}</span>
              </label>
            ))}
          </div>
          <div className="mt-4 space-y-3">
            {config.data_size.type === 'percentage' && (
              <input
                type="number"
                placeholder="Percentage (e.g., 80)"
                className="w-full p-3 border border-gray-300 rounded-lg"
                value={config.data_size.value ?? ''}
                onChange={(e) => setField('data_size', { type: 'percentage', value: parseInt(e.target.value) || 100 })}
              />
            )}
            {config.data_size.type === 'custom' && (
              <input
                type="number"
                placeholder="Number of rows"
                className="w-full p-3 border border-gray-300 rounded-lg"
                value={config.data_size.value ?? ''}
                onChange={(e) => setField('data_size', { type: 'custom', value: parseInt(e.target.value) || 1000 })}
              />
            )}
          </div>
        </div>
      </div>

      <ColumnSelection />
    </div>
  )
}
