export default function GenerationResultsSummary({ summary, entityLinkage }) {
  if (!summary) return null

  const totalRows = Object.values(summary).reduce((sum, table) => sum + table.rows, 0)
  const totalTables = Object.keys(summary).length

  return (
    <div id="generation-results">
      <h3 className="text-lg font-semibold mb-4">✨ Generation Summary</h3>
      <div className="bg-gray-50 rounded-lg p-6">
        {entityLinkage?.detected && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-4 text-sm text-blue-700">
            🔗 Cross-table consistency preserved via{' '}
            <code className="bg-blue-100 px-1 rounded">{entityLinkage.entity_key}</code> across{' '}
            {entityLinkage.master_table} and {entityLinkage.satellite_tables.join(', ')}.
          </div>
        )}
        <div className="grid md:grid-cols-3 gap-4">
          <div className="text-center">
            <h4 className="font-semibold">Tables Generated</h4>
            <p className="text-2xl font-bold text-blue-600">{totalTables}</p>
          </div>
          <div className="text-center">
            <h4 className="font-semibold">Total Rows</h4>
            <p className="text-2xl font-bold text-green-600">{totalRows}</p>
          </div>
          <div className="text-center">
            <h4 className="font-semibold">Status</h4>
            <p className="text-2xl font-bold text-purple-600">Success ✨</p>
          </div>
        </div>
        <div className="mt-4">
          <h4 className="font-semibold mb-2">Generated Tables:</h4>
          {Object.entries(summary).map(([tableName, data]) => (
            <div key={tableName} className="flex justify-between items-center py-2 border-b">
              <span>📊 {tableName}</span>
              <span className="text-green-600">
                {data.rows} rows, {data.columns} columns ({data.columns_synthesized} synthesized,{' '}
                {data.columns_copied} copied)
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
