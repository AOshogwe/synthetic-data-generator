import { useAppState } from '../state/AppContext'

const STEPS = [
  { id: 1, icon: 'fa-upload', label: 'Data Input' },
  { id: 2, icon: 'fa-cog', label: 'Configuration' },
  { id: 3, icon: 'fa-brain', label: 'Generation' },
  { id: 4, icon: 'fa-chart-line', label: 'Evaluation' },
  { id: 5, icon: 'fa-download', label: 'Export' },
]

export default function StepIndicator() {
  const { currentStep } = useAppState()

  return (
    <div className="bg-white shadow-sm">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between py-4">
          <div className="flex space-x-4 w-full">
            {STEPS.map((step) => {
              const cls =
                step.id < currentStep
                  ? 'step-completed'
                  : step.id === currentStep
                  ? 'step-active'
                  : 'bg-gray-200'
              return (
                <div key={step.id} className={`step-indicator ${cls} flex-1 text-center py-2 rounded-lg`}>
                  <i className={`fas ${step.icon} mb-1`}></i>
                  <div className="text-sm">{step.label}</div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
