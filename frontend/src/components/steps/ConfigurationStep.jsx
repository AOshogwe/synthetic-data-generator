import { useState } from 'react'
import { useAppDispatch } from '../../state/AppContext'
import GeneralSettingsTab from '../tabs/GeneralSettingsTab'
import PrivacyTab from '../tabs/PrivacyTab'
import RelationshipsTab from '../tabs/RelationshipsTab'
import AdvancedTab from '../tabs/AdvancedTab'

const TABS = [
  { id: 'general', label: '⚙️ General Settings', Component: GeneralSettingsTab },
  { id: 'privacy', label: '🛡️ Privacy & Anonymization', Component: PrivacyTab },
  { id: 'relationships', label: '🔗 Relationships', Component: RelationshipsTab },
  { id: 'advanced', label: '🚀 Advanced Options', Component: AdvancedTab },
]

export default function ConfigurationStep() {
  const dispatch = useAppDispatch()
  const [activeTab, setActiveTab] = useState('general')
  const ActiveComponent = TABS.find((t) => t.id === activeTab).Component

  return (
    <div className="bg-white rounded-xl shadow-lg p-8">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">
        <i className="fas fa-cog text-blue-600 mr-3"></i>Configuration
      </h2>

      <div className="flex flex-wrap border-b border-gray-200 mb-6">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-6 py-3 ${
              activeTab === tab.id ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-600 hover:text-blue-600'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <ActiveComponent />

      <div className="flex justify-between mt-8">
        <button
          onClick={() => dispatch({ type: 'SET_STEP', step: 1 })}
          className="bg-gray-600 text-white px-8 py-3 rounded-lg hover:bg-gray-700"
        >
          <i className="fas fa-arrow-left mr-2"></i>Previous
        </button>
        <button
          onClick={() => dispatch({ type: 'SET_STEP', step: 3 })}
          className="bg-blue-600 text-white px-8 py-3 rounded-lg hover:bg-blue-700"
        >
          Next: Generation <i className="fas fa-arrow-right ml-2"></i>
        </button>
      </div>
    </div>
  )
}
