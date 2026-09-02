import { useAppDispatch, useAppState } from '../state/AppContext'
import { COLUMN_ACTIONS } from '../config/contract'

export default function ColumnSelection() {
  const { uploadedData, config } = useAppState()
  const dispatch = useAppDispatch()

  if (!uploadedData) return null

  function setAction(tableName, columnName, action) {
    dispatch({ type: 'SET_COLUMN_ACTION', tableName, columnName, action })
  }

  return (
    <div className="mt-8">
      <h3 className="text-lg font-semibold mb-4">📋 Column Selection</h3>
      <div className="space-y-4">
        {Object.entries(uploadedData).map(([tableName, data]) => (
          <div key={tableName} className="bg-gray-50 rounded-lg p-4">
            <h4 className="font-semibold mb-3">📊 {tableName}</h4>
            <div className="grid md:grid-cols-2 gap-3">
              {data.column_names.map((column) => {
                const currentAction = config.column_selection[tableName]?.[column] ?? 'synthesize'
                const included = currentAction !== 'copy'
                return (
                  <label key={column} className="flex items-center">
                    <input
                      type="checkbox"
                      className="mr-3"
                      checked={included}
                      title="Uncheck to force this column to be copied unchanged, regardless of the dropdown"
                      onChange={(e) => setAction(tableName, column, e.target.checked ? 'synthesize' : 'copy')}
                    />
                    <span>{column}</span>
                    <select
                      className="ml-auto text-sm border rounded px-2 py-1"
                      value={currentAction}
                      onChange={(e) => setAction(tableName, column, e.target.value)}
                    >
                      {COLUMN_ACTIONS.map((opt) => (
                        <option key={opt.value} value={opt.value}>
                          {opt.label}
                        </option>
                      ))}
                    </select>
                  </label>
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
