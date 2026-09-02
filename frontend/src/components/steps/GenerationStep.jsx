import { useState } from 'react'
import { useAppDispatch, useAppState } from '../../state/AppContext'
import { configure, generate } from '../../api/client'
import { buildConfigPayload } from '../../config/contract'
import GenerationResultsSummary from '../GenerationResultsSummary'

export default function GenerationStep() {
  const { config, generationSummary, generationEntityLinkage } = useAppState()
  const dispatch = useAppDispatch()
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState({ pct: 0, text: '' })
  const [statuses, setStatuses] = useState({ loading: 'Pending', processing: 'Pending', generation: 'Pending' })
  const [error, setError] = useState(null)

  async function handleGenerate() {
    setBusy(true)
    setError(null)
    setProgress({ pct: 20, text: 'Configuring generation...' })
    setStatuses((s) => ({ ...s, loading: 'In Progress' }))
    try {
      const payload = buildConfigPayload(config)
      await configure(payload)

      setProgress({ pct: 40, text: 'Generating synthetic data...' })
      setStatuses((s) => ({ ...s, loading: 'Complete', processing: 'In Progress' }))

      const response = await generate()
      if (!response.success) throw new Error(response.error || 'Generation failed')

      setProgress({ pct: 100, text: 'Generation complete!' })
      setStatuses({ loading: 'Complete', processing: 'Complete', generation: 'Complete' })

      dispatch({
        type: 'SET_GENERATION_RESULT',
        summary: response.summary,
        entityLinkage: response.entity_linkage,
      })
    } catch (err) {
      setError(err.message)
      setStatuses((s) => ({ ...s, processing: 'Failed' }))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-8">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">
        <i className="fas fa-brain text-blue-600 mr-3"></i>Synthetic Data Generation
      </h2>

      {busy && (
        <div className="mb-6">
          <div className="bg-gray-200 rounded-full h-4 mb-4">
            <div className="progress-bar bg-blue-600 h-4 rounded-full" style={{ width: `${progress.pct}%` }}></div>
          </div>
          <div className="text-center text-gray-600 mb-6">{progress.text}</div>
        </div>
      )}

      <div className="grid md:grid-cols-3 gap-6 mb-8">
        <div className="bg-gray-50 rounded-lg p-6 text-center">
          <i className="fas fa-database text-3xl text-gray-400 mb-3"></i>
          <h3 className="font-semibold">Data Loading</h3>
          <p className="text-sm text-gray-600 mt-2">{statuses.loading}</p>
        </div>
        <div className="bg-gray-50 rounded-lg p-6 text-center">
          <i className="fas fa-cogs text-3xl text-gray-400 mb-3"></i>
          <h3 className="font-semibold">Processing</h3>
          <p className="text-sm text-gray-600 mt-2">{statuses.processing}</p>
        </div>
        <div className="bg-gray-50 rounded-lg p-6 text-center">
          <i className="fas fa-magic text-3xl text-gray-400 mb-3"></i>
          <h3 className="font-semibold">Generation</h3>
          <p className="text-sm text-gray-600 mt-2">{statuses.generation}</p>
        </div>
      </div>

      {error && <p className="text-sm text-red-600 mb-4">{error}</p>}

      <GenerationResultsSummary summary={generationSummary} entityLinkage={generationEntityLinkage} />

      <div className="flex justify-between mt-8">
        <button
          onClick={() => dispatch({ type: 'SET_STEP', step: 2 })}
          className="bg-gray-600 text-white px-8 py-3 rounded-lg hover:bg-gray-700"
        >
          <i className="fas fa-arrow-left mr-2"></i>Previous
        </button>
        <div className="space-x-4">
          <button
            onClick={handleGenerate}
            disabled={busy}
            className="bg-green-600 text-white px-8 py-3 rounded-lg hover:bg-green-700 disabled:bg-gray-400"
          >
            <i className="fas fa-play mr-2"></i>Start Generation
          </button>
          {generationSummary && (
            <button
              onClick={() => dispatch({ type: 'SET_STEP', step: 4 })}
              className="bg-blue-600 text-white px-8 py-3 rounded-lg hover:bg-blue-700"
            >
              Next: Evaluation <i className="fas fa-arrow-right ml-2"></i>
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
