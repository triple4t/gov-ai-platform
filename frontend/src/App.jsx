import React from 'react';
import Header from './components/Header';
import PRDPlatformPanel from './components/PRDPlatformPanel';

// --- Commented: full unified app (voice form, OCR, conversational, face, doc summary) ---
// import React, { useState, useCallback } from 'react';
// import VoicePanel from './components/VoicePanel';
// import FormPanel, { FORM_FIELDS } from './components/FormPanel';
// import OCRPanel from './components/OCRPanel';
// import DocumentSummarizationPanel from './components/DocumentSummarizationPanel';
// import ConversationalAIPanel from './components/ConversationalAIPanel';
// import FaceVerificationPanel from './components/FaceVerificationPanel';
// import { useVoiceRecorder } from './hooks/useVoiceRecorder';

/**
 * Lean UI: PRD Platform only (matches lean backend `app/main.py`).
 * Uncomment imports and the block below `PRDPlatformPanel` to restore the full app.
 */
export default function App() {
  return (
    <>
      <Header />
      <main className="main-content full-width">
        <PRDPlatformPanel />
      </main>
    </>
  );
}

/* Full app body (restore with Header activeTab + onTabChange):
  const [activeTab, setActiveTab] = useState(3);
  return (
    <>
      <Header activeTab={activeTab} onTabChange={setActiveTab} />
      <main className={`main-content ${activeTab !== 3 ? 'full-width' : ''}`}>
        {activeTab === 3 ? ( ... Voice + Form ... ) : activeTab === 4 ? <OCRPanel /> : ...}
      </main>
    </>
  );
*/
