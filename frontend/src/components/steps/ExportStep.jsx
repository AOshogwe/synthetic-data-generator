import { useState } from 'react'
import { useAppDispatch, useAppState } from '../../state/AppContext'
import { exportData } from '../../api/client'

const FORMATS = [
  { value: 'csv', label: '📊 CSV Files' },
  { value: 'excel', label: '📈 Excel Workbook' },
  { value: 'json', label: '🔗 JSON' },
  { value: 'parquet', label: '⚡ Parquet' },
]

function downloadBlob(blob, filename) {
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.style.display = 'none'
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  window.URL.revokeObjectURL(url)
  document.body.removeChild(a)
}

export default function ExportStep() {
  const { evaluationResults } = useAppState()
  const dispatch = useAppDispatch()
  const [format, setFormat] = useState('csv')
  const [includeMetadata, setIncludeMetadata] = useState(true)
  const [includeSchema, setIncludeSchema] = useState(true)
  const [includeEvaluation, setIncludeEvaluation] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [status, setStatus] = useState(null)

  async function handleExport() {
    setBusy(true)
    setError(null)
    setStatus('Preparing export...')
    try {
      const blob = await exportData({ format, includeMetadata, includeSchema, includeEvaluation })
      const timestamp = new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-')
      downloadBlob(blob, `synthetic_data_${timestamp}.zip`)
      setStatus('Export complete ✅')
    } catch (err) {
      setError(err.message)
      setStatus(null)
    } finally {
      setBusy(false)
    }
  }

  function handleDownloadReport() {
    if (!evaluationResults) {
      setError('Generate an evaluation report first (previous step) before downloading it.')
      return
    }
    const rows = Object.entries(evaluationResults)
      .map(
        ([table, r]) =>
          `<tr><td>${table}</td><td>${(r.statistical_similarity ?? 0).toFixed(3)}</td><td>${(r.privacy_score ?? 0).toFixed(
            3
          )}</td><td>${(r.utility_score ?? 0).toFixed(3)}</td></tr>`
      )
      .join('')
    const html = `<!doctype html><html><head><meta charset="utf-8"><title>Synthetic Data Report</title>
      <style>body{font-family:sans-serif;padding:2rem} table{border-collapse:collapse;width:100%} td,th{border:1px solid #ddd;padding:8px}</style>
      </head><body><h1>Synthetic Data Evaluation Report</h1><p>Generated ${new Date().toLocaleString()}</p>
      <table><tr><th>Table</th><th>Statistical Similarity</th><th>Privacy Score</th><th>Utility Score</th></tr>${rows}</table>
      </body></html>`
    downloadBlob(new Blob([html], { type: 'text/html' }), `synthetic_data_report_${Date.now()}.html`)
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-8">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">
        <i className="fas fa-download text-blue-600 mr-3"></i>Export Synthetic Data
      </h2>

      <div className="grid md:grid-cols-2 gap-8">
        <div className="config-panel rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">📄 Export Format</h3>
          <div className="space-y-3">
            {FORMATS.map((f) => (
              <label key={f.value} className="flex items-center">
                <input
                  type="radio"
                  name="export-format"
                  className="mr-3"
                  checked={format === f.value}
                  onChange={() => setFormat(f.value)}
                />
                <span>{f.label}</span>
              </label>
            ))}
          </div>
        </div>

        <div className="config-panel rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">⚙️ Export Options</h3>
          <div className="space-y-3">
            <label className="flex items-center">
              <input type="checkbox" className="mr-3" checked={includeMetadata} onChange={(e) => setIncludeMetadata(e.target.checked)} />
              <span>Include metadata</span>
            </label>
            <label className="flex items-center">
              <input type="checkbox" className="mr-3" checked={includeSchema} onChange={(e) => setIncludeSchema(e.target.checked)} />
              <span>Include schema information</span>
            </label>
            <label className="flex items-center">
              <input
                type="checkbox"
                className="mr-3"
                checked={includeEvaluation}
                onChange={(e) => setIncludeEvaluation(e.target.checked)}
              />
              <span>Include evaluation report</span>
            </label>
          </div>
        </div>
      </div>

      {status && <p className="text-sm text-green-600 mt-6">{status}</p>}
      {error && <p className="text-sm text-red-600 mt-6">{error}</p>}

      <div className="mt-8 text-center space-x-4">
        <button
          onClick={handleExport}
          disabled={busy}
          className="bg-green-600 text-white px-8 py-3 rounded-lg hover:bg-green-700 disabled:bg-gray-400"
        >
          <i className="fas fa-download mr-2"></i>Download Synthetic Data
        </button>
        <button onClick={handleDownloadReport} className="bg-purple-600 text-white px-8 py-3 rounded-lg hover:bg-purple-700">
          <i className="fas fa-file-pdf mr-2"></i>Download Report
        </button>
      </div>

      <div className="flex justify-between mt-8">
        <button
          onClick={() => dispatch({ type: 'SET_STEP', step: 4 })}
          className="bg-gray-600 text-white px-8 py-3 rounded-lg hover:bg-gray-700"
        >
          <i className="fas fa-arrow-left mr-2"></i>Previous
        </button>
        <button onClick={() => dispatch({ type: 'RESET' })} className="bg-blue-600 text-white px-8 py-3 rounded-lg hover:bg-blue-700">
          <i className="fas fa-redo mr-2"></i>Start Over
        </button>
      </div>
    </div>
  )
}
