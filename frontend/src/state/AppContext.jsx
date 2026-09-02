import { createContext, useContext, useMemo, useReducer } from 'react'
import { DEFAULT_CONFIG } from '../config/contract'

const initialState = {
  currentStep: 1,
  uploadedData: null, // { [tableName]: { rows, columns, column_names, sample_data, data_types } }
  entityLinkage: null,
  config: structuredClone(DEFAULT_CONFIG),
  generationSummary: null,
  generationEntityLinkage: null,
  evaluationResults: null,
  status: 'Ready',
  error: null,
}

function reducer(state, action) {
  switch (action.type) {
    case 'SET_STEP':
      return { ...state, currentStep: action.step }
    case 'SET_UPLOADED_DATA':
      return {
        ...state,
        uploadedData: action.tables,
        entityLinkage: action.entityLinkage || null,
        // Reset column_selection to "synthesize everything" for the newly
        // uploaded tables -- this mirrors the old UI's default DOM state
        // (checkbox checked, dropdown defaulted to "Synthesize") which is
        // what made task #32's fix work: the real default is "synthesize",
        // not silently "copy".
        config: {
          ...state.config,
          column_selection: Object.fromEntries(
            Object.entries(action.tables).map(([tableName, data]) => [
              tableName,
              Object.fromEntries(data.column_names.map((col) => [col, 'synthesize'])),
            ])
          ),
        },
      }
    case 'SET_COLUMN_ACTION': {
      const { tableName, columnName, action: colAction } = action
      return {
        ...state,
        config: {
          ...state.config,
          column_selection: {
            ...state.config.column_selection,
            [tableName]: {
              ...state.config.column_selection[tableName],
              [columnName]: colAction,
            },
          },
        },
      }
    }
    case 'SET_CONFIG_FIELD':
      return { ...state, config: { ...state.config, [action.field]: action.value } }
    case 'SET_GENERATION_RESULT':
      return {
        ...state,
        generationSummary: action.summary,
        generationEntityLinkage: action.entityLinkage || null,
      }
    case 'SET_EVALUATION_RESULTS':
      return { ...state, evaluationResults: action.results }
    case 'SET_STATUS':
      return { ...state, status: action.status, error: null }
    case 'SET_ERROR':
      return { ...state, error: action.error }
    case 'RESET':
      return structuredClone(initialState)
    default:
      throw new Error(`Unknown action type: ${action.type}`)
  }
}

const AppStateContext = createContext(null)
const AppDispatchContext = createContext(null)

export function AppProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, undefined, () => structuredClone(initialState))
  const value = useMemo(() => state, [state])
  return (
    <AppStateContext.Provider value={value}>
      <AppDispatchContext.Provider value={dispatch}>{children}</AppDispatchContext.Provider>
    </AppStateContext.Provider>
  )
}

export function useAppState() {
  const ctx = useContext(AppStateContext)
  if (!ctx) throw new Error('useAppState must be used within AppProvider')
  return ctx
}

export function useAppDispatch() {
  const ctx = useContext(AppDispatchContext)
  if (!ctx) throw new Error('useAppDispatch must be used within AppProvider')
  return ctx
}
