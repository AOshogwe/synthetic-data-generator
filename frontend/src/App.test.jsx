import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'
import App from './App'

// Realistic response shapes -- copied from actual /api/upload, /api/configure,
// and /api/generate responses observed against the real Flask backend during
// this session's manual testing (tasks #22, #30-34), not invented shapes.
const UPLOAD_RESPONSE = {
  success: true,
  files_uploaded: 1,
  session_id: '20260830_000000',
  schema: {
    mock_table: { columns: { 'First Name': {}, 'Last name': {}, Identifier: {}, 'Age (yrs)': {} } },
  },
  tables: {
    mock_table: {
      rows: 3,
      columns: 4,
      column_names: ['First Name', 'Last name', 'Identifier', 'Age (yrs)'],
      sample_data: [{ 'First Name': 'Sidney', 'Last name': 'Crosby', Identifier: 'T001', 'Age (yrs)': 16 }],
      data_types: { 'First Name': 'object', 'Last name': 'object', Identifier: 'object', 'Age (yrs)': 'int64' },
    },
  },
  entity_linkage: { detected: false },
}

const CONFIGURE_RESPONSE = {
  success: true,
  message: 'Advanced configuration applied with column selection',
  column_selection_applied: true,
  features_enabled: {
    name_anonymization: false,
    age_grouping: false,
    address_anonymization: false,
    relationship_preservation: true,
    perturbation_mode: false,
    privacy_level: 'balanced',
    differential_privacy: false,
  },
}

const GENERATE_RESPONSE = {
  success: true,
  message: 'Data processed successfully using copy_with_privacy_features',
  method_used: 'copy_with_privacy_features',
  summary: {
    mock_table: {
      rows: 3,
      columns: 4,
      original_rows: 3,
      sample_data: [{ 'First Name': 'Sidney', 'Last name': 'Crosby', Identifier: 'T001', 'Age (yrs)': 16 }],
      generation_method: 'copy_with_privacy',
      generation_time: 0.01,
      columns_synthesized: 1,
      columns_copied: 3,
      synthesized_column_names: ['Identifier'],
      copied_column_names: ['First Name', 'Last name', 'Age (yrs)'],
    },
  },
  generation_time: 0.01,
  total_original_rows: 3,
  total_synthetic_rows: 3,
  method_used: 'copy_with_privacy_features',
  synthesis_required: true,
  entity_linkage: { detected: false },
}

vi.mock('./api/client', () => ({
  uploadFiles: vi.fn(() => Promise.resolve(UPLOAD_RESPONSE)),
  configure: vi.fn(() => Promise.resolve(CONFIGURE_RESPONSE)),
  generate: vi.fn(() => Promise.resolve(GENERATE_RESPONSE)),
  evaluate: vi.fn(),
  debugColumns: vi.fn(),
  exportData: vi.fn(),
}))

import { configure, uploadFiles } from './api/client'

beforeEach(() => {
  vi.clearAllMocks()
})

async function uploadOneFile() {
  const file = new File(['a,b\n1,2'], 'mock.csv', { type: 'text/csv' })
  const input = document.querySelector('input[type="file"]')
  await fireEvent.change(input, { target: { files: [file] } })
  await waitFor(() => expect(screen.getByText(/mock_table/)).toBeInTheDocument())
}

describe('full upload -> configure -> generate flow', () => {
  it('uploads a file and shows the data preview', async () => {
    render(<App />)
    await uploadOneFile()
    expect(screen.getByText(/3 rows, 4 columns/)).toBeInTheDocument()
    expect(uploadFiles).toHaveBeenCalledTimes(1)
  })

  it('advances to Configuration and shows Column Selection populated from the upload', async () => {
    render(<App />)
    await uploadOneFile()
    fireEvent.click(screen.getByText(/Next: Configuration/))
    expect(await screen.findByText('📋 Column Selection')).toBeInTheDocument()
    expect(screen.getByText('Identifier')).toBeInTheDocument()
    expect(screen.getByText('First Name')).toBeInTheDocument()
  })

  it('setting a column to Copy and enabling the global name toggle sends both in the payload (task #34 rule surfaced, not hidden)', async () => {
    render(<App />)
    await uploadOneFile()
    fireEvent.click(screen.getByText(/Next: Configuration/))
    await screen.findByText('📋 Column Selection')

    // Set "First Name" column to Copy original
    const firstNameLabel = screen.getByText('First Name').closest('label')
    const select = within(firstNameLabel).getByRole('combobox')
    fireEvent.change(select, { target: { value: 'copy' } })

    // Switch to Privacy tab and enable the global anonymize-names toggle
    fireEvent.click(screen.getByText(/Privacy & Anonymization/))
    const nameToggle = screen.getByText('Anonymize name columns').closest('label').querySelector('input')
    fireEvent.click(nameToggle)

    // Go generate
    fireEvent.click(screen.getByText(/Next: Generation/))
    fireEvent.click(screen.getByText(/Start Generation/))

    await waitFor(() => expect(configure).toHaveBeenCalledTimes(1))
    const payload = configure.mock.calls[0][0]
    expect(payload.anonymize_names).toBe(true)
    expect(payload.column_selection.mock_table['First Name']).toBe('copy')
    // The backend (tested separately, task #34) is what actually enforces
    // that 'copy' wins -- this test only proves the UI sends both facts
    // faithfully rather than one silently overwriting the other client-side.
  })

  it('shows the generation results summary after Start Generation', async () => {
    render(<App />)
    await uploadOneFile()
    fireEvent.click(screen.getByText(/Next: Configuration/))
    await screen.findByText('📋 Column Selection')
    fireEvent.click(screen.getByText(/Next: Generation/))
    fireEvent.click(screen.getByText(/Start Generation/))

    expect(await screen.findByText('✨ Generation Summary')).toBeInTheDocument()
    expect(screen.getByText(/1 synthesized/)).toBeInTheDocument()
  })
})
