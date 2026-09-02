import { useRef, useState } from 'react'
import { useAppDispatch, useAppState } from '../../state/AppContext'
import { uploadFiles } from '../../api/client'
import EntityLinkageBanner from '../EntityLinkageBanner'

export default function DataInputStep() {
  const { uploadedData, entityLinkage } = useAppState()
  const dispatch = useAppDispatch()
  const fileInputRef = useRef(null)
  const [fileNames, setFileNames] = useState([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  async function handleFiles(fileList) {
    const files = Array.from(fileList)
    if (files.length === 0) return
    setFileNames(files.map((f) => f.name))
    setBusy(true)
    setError(null)
    dispatch({ type: 'SET_STATUS', status: 'Uploading files...' })
    try {
      const response = await uploadFiles(files)
      if (!response.success) throw new Error(response.error || 'Upload failed')
      dispatch({
        type: 'SET_UPLOADED_DATA',
        tables: response.tables,
        entityLinkage: response.entity_linkage,
      })
      dispatch({ type: 'SET_STATUS', status: 'Files uploaded successfully' })
    } catch (err) {
      setError(err.message)
      dispatch({ type: 'SET_ERROR', error: err.message })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-8">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">
        <i className="fas fa-upload text-blue-600 mr-3"></i>Data Input
      </h2>

      <div className="grid md:grid-cols-2 gap-8">
        <div className="card-hover bg-gray-50 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">📁 Upload CSV/Excel Files</h3>
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".csv,.xlsx,.json"
              className="hidden"
              onChange={(e) => handleFiles(e.target.files)}
            />
            <label onClick={() => fileInputRef.current?.click()} className="cursor-pointer">
              <i className="fas fa-cloud-upload-alt text-4xl text-gray-400 mb-4"></i>
              <p className="text-gray-600">{busy ? 'Uploading...' : 'Click to upload files or drag and drop'}</p>
              <p className="text-sm text-gray-500 mt-2">Supports CSV, Excel, JSON (Max 100MB)</p>
            </label>
          </div>
          {fileNames.length > 0 && (
            <div className="mt-4 space-y-1 text-sm text-gray-600">
              {fileNames.map((name) => (
                <div key={name}>📄 {name}</div>
              ))}
            </div>
          )}
          {error && <p className="text-sm text-red-600 mt-2">{error}</p>}
        </div>

        <div className="card-hover bg-gray-50 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">🗄️ Database Connection</h3>
          <div className="space-y-4 opacity-60">
            <select disabled className="w-full p-3 border border-gray-300 rounded-lg">
              <option>Select Database Type</option>
            </select>
            <input disabled placeholder="Connection string" className="w-full p-3 border border-gray-300 rounded-lg" />
            <button disabled className="w-full bg-gray-400 text-white py-3 rounded-lg cursor-not-allowed">
              <i className="fas fa-plug mr-2"></i>Coming soon
            </button>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Not implemented yet -- the previous UI's version of this only simulated a connection rather than
            performing one, so it isn't carried over here.
          </p>
        </div>
      </div>

      {uploadedData && (
        <div className="mt-8">
          <h3 className="text-lg font-semibold mb-4">📊 Data Preview</h3>
          <div className="bg-gray-50 rounded-lg p-4 overflow-x-auto">
            <EntityLinkageBanner entityLinkage={entityLinkage} />
            <div className="grid gap-4">
              {Object.entries(uploadedData).map(([tableName, data]) => (
                <div key={tableName} className="bg-white rounded-lg p-4 border">
                  <h4 className="font-semibold mb-2">📊 {tableName}</h4>
                  <p className="text-sm text-gray-600">
                    {data.rows} rows, {data.columns} columns
                  </p>
                  <p className="text-sm text-gray-600">
                    Columns: {data.column_names.slice(0, 5).join(', ')}
                    {data.column_names.length > 5 ? '...' : ''}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      <div className="flex justify-end mt-8">
        <button
          onClick={() => dispatch({ type: 'SET_STEP', step: 2 })}
          disabled={!uploadedData}
          className="bg-blue-600 text-white px-8 py-3 rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
        >
          Next: Configuration <i className="fas fa-arrow-right ml-2"></i>
        </button>
      </div>
    </div>
  )
}
