import React, { useState, useCallback } from 'react';
import Header from './components/Header';
import VoicePanel from './components/VoicePanel';
import FormPanel, { FORM_FIELDS } from './components/FormPanel';
import OCRPanel from './components/OCRPanel';
import DocumentSummarizationPanel from './components/DocumentSummarizationPanel';
import ConversationalAIPanel from './components/ConversationalAIPanel';
import FaceVerificationPanel from './components/FaceVerificationPanel';
import { useVoiceRecorder } from './hooks/useVoiceRecorder';

/**
 * App - Main application component.
 * Orchestrates the GUIDED field-by-field voice form filling.
 * The user fills one field at a time: record → extract → fill → next field.
 */
export default function App() {
  const [language, setLanguage] = useState('hi-IN');
  const [formType, setFormType] = useState('aadhaar');
  const [activeFieldIndex, setActiveFieldIndex] = useState(0);
  const [formValues, setFormValues] = useState({
    full_name: '',
    phone_number: '',
    pan_card: '',
    age: '',
    aadhaar_number: '',
    address: '',
  });

  const {
    isRecording,
    isProcessing,
    transcript,
    translatedText,
    status,
    startRecording,
    stopRecordingForField,
    reset: resetVoice,
    setStatus,
  } = useVoiceRecorder();

  /**
   * Handle mic button click in guided mode:
   * - First click: start recording
   * - Second click: stop recording, extract the value for the active field,
   *   fill it, and advance to the next empty field
   */
  const handleMicClick = useCallback(async () => {
    if (isRecording) {
      // Stop recording and extract the value for the active field
      const activeField = FORM_FIELDS[activeFieldIndex];
      if (!activeField) return;

      const result = await stopRecordingForField(
        language,
        activeField.key,
        activeField.labelEn
      );

      if (result && result.value) {
        let cleanedValue = result.value;

        // Robust client-side cleaning for numeric/strict format fields
        const wordToNum = {
          'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
          'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
          'ten': '10', 'twenty': '20', 'thirty': '30', 'forty': '40', 'fifty': '50',
          'sixty': '60', 'seventy': '70', 'eighty': '80', 'ninety': '90',
          'shunya': '0', 'ek': '1', 'do': '2', 'teen': '3', 'char': '4',
          'paanch': '5', 'chhah': '6', 'saat': '7', 'aath': '8', 'nau': '9',
          'shunnya': '0', 'don': '2', 'tin': '3', 'pach': '5', 'saha': '6', 'sat': '7', 'ath': '8', 'nav': '9',
          'शून्य': '0', 'एक': '1', 'दो': '2', 'तीन': '3', 'चार': '4',
          'पांच': '5', 'छह': '6', 'सात': '7', 'आठ': '8', 'नौ': '9',
          'दोन': '2', 'पाच': '5', 'सहा': '6', 'नऊ': '9'
        };
        if (['phone_number', 'aadhaar_number', 'pan_card'].includes(activeField.key)) {
          let formattedStr = cleanedValue.toLowerCase();
          for (const [word, digit] of Object.entries(wordToNum)) {
            formattedStr = formattedStr.replace(new RegExp(word, 'g'), digit);
          }
          if (activeField.key === 'pan_card') {
            cleanedValue = formattedStr.replace(/[^a-z0-9]/gi, '').toUpperCase();
          } else {
            cleanedValue = formattedStr.replace(/\D/g, '');
          }
        } else if (activeField.key === 'age') {
          // Age: always show as number (e.g. "Twenty one." -> "21"). Keep DD/MM/YYYY if it looks like a date.
          if (/^\d{1,2}\/\d{1,2}\/\d{2,4}$/.test(cleanedValue.trim())) {
            cleanedValue = cleanedValue.trim();
          } else {
            let formattedStr = cleanedValue.toLowerCase().replace(/[.\s]+/g, ' ');
            for (const [word, digit] of Object.entries(wordToNum)) {
              formattedStr = formattedStr.replace(new RegExp('\\b' + word + '\\b', 'gi'), digit);
            }
            const groups = formattedStr.match(/\d+/g);
            if (groups && groups.length === 1) {
              const num = parseInt(groups[0], 10);
              if (num > 0 && num < 150) cleanedValue = String(num);
            } else if (groups && groups.length === 2) {
              const a = parseInt(groups[0], 10), b = parseInt(groups[1], 10);
              if (a >= 10 && a <= 90 && a % 10 === 0 && b >= 0 && b <= 9) {
                const num = a + b;
                if (num > 0 && num < 150) cleanedValue = String(num);
              }
            }
          }
        }

        if (cleanedValue.trim() === '') {
          // If extraction returned empty, stay on the same field and warn user
          setStatus({
            type: 'error',
            message: `⚠️ Could not hear a valid ${activeField.labelEn}. Please try again.`,
          });
          return;
        }

        // Fill the active field
        setFormValues((prev) => ({
          ...prev,
          [activeField.key]: cleanedValue,
        }));

        // Advance to the next empty field
        let nextIndex = activeFieldIndex + 1;
        // Search sequentially for the next empty field
        while (nextIndex < FORM_FIELDS.length) {
          const nextKey = FORM_FIELDS[nextIndex].key;
          // Check current formValues, but use the new value if we just filled it
          const currentValue = activeField.key === nextKey ? cleanedValue : formValues[nextKey];
          if (!currentValue || currentValue.trim() === '') {
            break; // Found an empty field!
          }
          nextIndex++;
        }

        if (nextIndex < FORM_FIELDS.length) {
          setActiveFieldIndex(nextIndex);
          const nextField = FORM_FIELDS[nextIndex];
          setStatus({
            type: 'info',
            message: `✅ ${activeField.labelEn} filled! Next: ${nextField.labelEn}`,
          });
        } else {
          setActiveFieldIndex(FORM_FIELDS.length); // All done
          setStatus({
            type: 'success',
            message: '🎉 All 6 fields have been filled successfully!',
          });
        }
      }
    } else {
      // Start recording for the current field
      await startRecording(language);
    }
  }, [isRecording, activeFieldIndex, language, formValues, startRecording, stopRecordingForField, setStatus]);

  /**
   * Handle individual form field manual edit.
   */
  const handleFormValueChange = useCallback((key, value) => {
    setFormValues((prev) => ({ ...prev, [key]: value }));
  }, []);

  /**
   * Calculate how many fields are currently filled.
   */
  const fieldsFilled = Object.values(formValues).filter(
    (v) => v && v.trim() !== ''
  ).length;

  /**
   * Reset everything.
   */
  const handleReset = useCallback(() => {
    setFormValues({
      full_name: '',
      phone_number: '',
      pan_card: '',
      age: '',
      aadhaar_number: '',
      address: '',
    });
    setActiveFieldIndex(0);
    resetVoice();
  }, [resetVoice]);

  /**
   * Handle form submission.
   */
  const handleSubmit = useCallback(() => {
    const filledFields = Object.entries(formValues).filter(
      ([, v]) => v && v.trim() !== ''
    );
    if (filledFields.length === 0) {
      alert('Please fill at least one field before submitting.');
      return;
    }
    console.log('Form submitted:', formValues);
    alert(
      `Form submitted successfully!\n\n${filledFields
        .map(([k, v]) => `${k}: ${v}`)
        .join('\n')}`
    );
  }, [formValues]);

  const [activeTab, setActiveTab] = useState(3);

  return (
    <>
      <Header activeTab={activeTab} onTabChange={setActiveTab} />

    <main className={`main-content ${activeTab !== 3 ? 'full-width' : ''}`}>
      {activeTab === 3 ? (
        <>
          <VoicePanel
            isRecording={isRecording}
            isProcessing={isProcessing}
            transcript={transcript}
            translatedText={translatedText}
            fieldsFilled={fieldsFilled}
            status={status}
            language={language}
            onLanguageChange={setLanguage}
            onMicClick={handleMicClick}
            activeFieldIndex={activeFieldIndex < FORM_FIELDS.length ? activeFieldIndex : null}
          />

          <FormPanel
            formType={formType}
            onFormTypeChange={setFormType}
            formValues={formValues}
            onFormValueChange={handleFormValueChange}
            onReset={handleReset}
            onSubmit={handleSubmit}
            activeFieldIndex={activeFieldIndex < FORM_FIELDS.length ? activeFieldIndex : null}
            onFieldClick={(index) => {
              if (!isRecording && !isProcessing) {
                setActiveFieldIndex(index);
                setStatus({
                  type: 'info',
                  message: `Switched to: ${FORM_FIELDS[index].labelEn}`,
                });
              }
            }}
          />
        </>
      ) : activeTab === 4 ? (
        <OCRPanel />
      ) : activeTab === 5 ? (
        <FaceVerificationPanel />
      ) : activeTab === 1 ? (
        <DocumentSummarizationPanel />
      ) : activeTab === 2 ? (
        <ConversationalAIPanel />
      ) : (
        <div className="glass-panel" style={{ gridColumn: 'span 2', padding: '100px', textAlign: 'center', width: '100%' }}>
          <h2>Service Unavailable</h2>
          <p>This AI service is currently in development.</p>
        </div>
      )}
    </main>
    </>
  );
}
