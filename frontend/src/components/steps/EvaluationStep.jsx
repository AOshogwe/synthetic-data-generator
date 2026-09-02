import { useState } from 'react'
import { useAppDispatch, useAppState } from '../../state/AppContext'
import { evaluate } from '../../api/client'

function riskLabel(avgPrivacy, thresholds) {
  if (avgPrivacy > 0.8) return thresholds[0]
  if (avgPrivacy > 0.6) return thresholds[1]
  return thresholds[2]
}

export default function EvaluationStep() {
  const { evaluationResults } = useAppState()
  const dispatch = useAppDispatch()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  async function handleGenerateReport() {
    setBusy(true)
    setError(null)
    try {
      const response = await evaluate()
      if (!response.success) throw new Error(response.error || 'Evaluation failed')
      dispatch({ type: 'SET_EVALUATION_RESULTS', results: response.evaluation_results })
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  let similarity = null,
    privacy = null,
    utility = null
  if (evaluationResults) {
    const tables = Object.values(evaluationResults)
    const n = tables.length || 1
    similarity = tables.reduce((s, t) => s + (t.statistical_similarity || 0), 0) / n
    privacy = tables.reduce((s, t) => s + (t.privacy_score || 0), 0) / n
    utility = tables.reduce((s, t) => s + (t.utility_score || 0), 0) / n
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-8">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">
        <i className="fas fa-chart-line text-blue-600 mr-3"></i>Data Quality Evaluation
      </h2>

      <div className="grid md:grid-cols-3 gap-6 mb-8">
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-semibold text-blue-800">Statistical Similarity</h3>
              <p className="text-2xl font-bold text-blue-600">{similarity !== null ? similarity.toFixed(3) : '--'}</p>
            </div>
            <i className="fas fa-chart-bar text-3xl text-blue-400"></i>
          </div>
          <p className="text-sm text-blue-600 mt-2">How well synthetic data matches original distributions</p>
        </div>

        <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-semibold text-green-800">Privacy Protection</h3>
              <p className="text-2xl font-bold text-green-600">{privacy !== null ? privacy.toFixed(3) : '--'}</p>
            </div>
            <i className="fas fa-shield-alt text-3xl text-green-400"></i>
          </div>
          <p className="text-sm text-green-600 mt-2">Level of privacy protection achieved</p>
        </div>

        <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-semibold text-purple-800">ML Utility</h3>
              <p className="text-2xl font-bold text-purple-600">{utility !== null ? utility.toFixed(3) : '--'}</p>
            </div>
            <i className="fas fa-robot text-3xl text-purple-400"></i>
          </div>
          <p className="text-sm text-purple-600 mt-2">How useful for machine learning tasks</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-gray-50 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">🔒 Privacy Analysis</h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span>Membership Inference Risk</span>
              <span className="font-semibold">{privacy !== null ? riskLabel(privacy, ['Low', 'Medium', 'High']) : '--'}</span>
            </div>
            <div className="flex justify-between items-center">
              <span>Attribute Disclosure Risk</span>
              <span className="font-semibold">
                {privacy !== null ? riskLabel(privacy, ['Very Low', 'Low', 'Medium']) : '--'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span>Re-identification Risk</span>
              <span className="font-semibold">{privacy !== null ? riskLabel(privacy, ['Low', 'Medium', 'High']) : '--'}</span>
            </div>
          </div>
        </div>

        {utility !== null && (
          <div className="bg-gray-50 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4">📊 Data Quality Report</h3>
            <div className="space-y-4">
              <div className="flex justify-between">
                <span>Data completeness</span>
                <span className="font-semibold text-green-600">{(utility * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span>Relationship preservation</span>
                <span className="font-semibold text-blue-600">{(similarity * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span>Distribution similarity</span>
                <span className="font-semibold text-blue-600">{(similarity * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {error && <p className="text-sm text-red-600 mt-4">{error}</p>}

      <div className="flex justify-between mt-8">
        <button
          onClick={() => dispatch({ type: 'SET_STEP', step: 3 })}
          className="bg-gray-600 text-white px-8 py-3 rounded-lg hover:bg-gray-700"
        >
          <i className="fas fa-arrow-left mr-2"></i>Previous
        </button>
        <div className="space-x-4">
          <button
            onClick={handleGenerateReport}
            disabled={busy}
            className="bg-purple-600 text-white px-8 py-3 rounded-lg hover:bg-purple-700 disabled:bg-gray-400"
          >
            <i className="fas fa-file-alt mr-2"></i>Generate Report
          </button>
          <button
            onClick={() => dispatch({ type: 'SET_STEP', step: 5 })}
            className="bg-blue-600 text-white px-8 py-3 rounded-lg hover:bg-blue-700"
          >
            Next: Export <i className="fas fa-arrow-right ml-2"></i>
          </button>
        </div>
      </div>
    </div>
  )
}
