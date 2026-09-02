export default function EntityLinkageBanner({ entityLinkage }) {
  if (!entityLinkage) return null

  if (entityLinkage.detected) {
    return (
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
        <p className="font-semibold text-blue-800">🔗 Linked tables detected</p>
        <p className="text-sm text-blue-700 mt-1">
          Shared identifier <code className="bg-blue-100 px-1 rounded">{entityLinkage.entity_key}</code> found in{' '}
          <strong>{entityLinkage.master_table}</strong> ({entityLinkage.entity_count} entities) and{' '}
          {entityLinkage.satellite_tables.length} linked table{entityLinkage.satellite_tables.length === 1 ? '' : 's'}:{' '}
          {entityLinkage.satellite_tables.join(', ')}.
        </p>
        <p className="text-sm text-blue-700 mt-1">
          Each synthetic entity's rows will stay consistent across all of these tables during generation.
        </p>
      </div>
    )
  }

  if (entityLinkage.candidate_tables) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
        <p className="font-semibold text-yellow-800">⚠️ Possible shared identifier not linkable</p>
        <p className="text-sm text-yellow-700 mt-1">
          Found <code className="bg-yellow-100 px-1 rounded">{entityLinkage.entity_key}</code> in{' '}
          {entityLinkage.candidate_tables.join(', ')}, but {entityLinkage.reason}
        </p>
      </div>
    )
  }

  return null
}
