import { AppProvider, useAppState } from './state/AppContext'
import StepIndicator from './components/StepIndicator'
import DataInputStep from './components/steps/DataInputStep'
import ConfigurationStep from './components/steps/ConfigurationStep'
import GenerationStep from './components/steps/GenerationStep'
import EvaluationStep from './components/steps/EvaluationStep'
import ExportStep from './components/steps/ExportStep'

const STEP_COMPONENTS = {
  1: DataInputStep,
  2: ConfigurationStep,
  3: GenerationStep,
  4: EvaluationStep,
  5: ExportStep,
}

function Shell() {
  const { currentStep, status } = useAppState()
  const StepComponent = STEP_COMPONENTS[currentStep]

  return (
    <>
      <header className="gradient-bg text-white shadow-lg">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold">🔮 Synthetic Data Generator</h1>
              <p className="text-blue-100 mt-1">Advanced privacy-preserving synthetic data generation</p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="bg-white bg-opacity-20 rounded-lg px-4 py-2">
                <span className="text-sm">Status: {status}</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <StepIndicator />

      <div className="container mx-auto px-4 py-8">
        <div className="fade-in">
          <StepComponent />
        </div>
      </div>

      <footer className="bg-gray-800 text-white py-8 mt-16">
        <div className="container mx-auto px-4 text-center">
          <p>&copy; 2024 Synthetic Data Generator. Built with privacy and utility in mind.</p>
          <p className="text-gray-400 mt-2">Advanced privacy-preserving synthetic data generation platform</p>
        </div>
      </footer>
    </>
  )
}

export default function App() {
  return (
    <AppProvider>
      <div className="bg-gray-50 min-h-screen">
        <Shell />
      </div>
    </AppProvider>
  )
}
