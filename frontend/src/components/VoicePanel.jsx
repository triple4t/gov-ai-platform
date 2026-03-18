import React from 'react';
import { FORM_FIELDS } from './FormPanel';

/**
 * VoicePanel - Left side of the UI.
 * Shows the active field prompt, microphone, language selector,
 * transcript, and form completion progress.
 */
export default function VoicePanel({
  isRecording,
  isProcessing,
  transcript,
  translatedText,
  fieldsFilled,
  status,
  language,
  onLanguageChange,
  onMicClick,
  activeFieldIndex,
}) {
  const LANGUAGES = [
    { code: 'hi-IN', label: 'हिन्दी', labelEn: 'Hindi' },
    { code: 'mr-IN', label: 'मराठी', labelEn: 'Marathi' },
    { code: 'en-IN', label: 'English', labelEn: 'English' },
  ];

  const progressPercent = (fieldsFilled / 6) * 100;
  const activeField = activeFieldIndex !== null && activeFieldIndex < FORM_FIELDS.length
    ? FORM_FIELDS[activeFieldIndex]
    : null;
  const allDone = fieldsFilled >= 6;

  return (
    <div className="voice-panel">
      {/* --- Voice Input Card --- */}
      <div className="voice-card" id="voice-card">
        <h3>Voice Input</h3>

        {/* --- Active Field Prompt --- */}
        {activeField && !allDone && (
          <div className="field-prompt fade-in" id="field-prompt">
            <div className="prompt-step">
              Step {activeFieldIndex + 1} of 6
            </div>
            <div className="prompt-icon">{activeField.icon}</div>
            <div className="prompt-text">
              {activeField.prompt}
            </div>
            <div className="prompt-text-hi">
              {activeField.promptHi}
            </div>
          </div>
        )}

        {allDone && (
          <div className="field-prompt fade-in done">
            <div className="prompt-icon">🎉</div>
            <div className="prompt-text">All fields filled!</div>
            <div className="prompt-text-hi">सभी फ़ील्ड भर दी गई हैं!</div>
          </div>
        )}

        {/* Microphone Button */}
        <div className="mic-btn-wrapper">
          <button
            className={`mic-btn${isRecording ? ' recording' : ''}${allDone ? ' done' : ''}`}
            onClick={onMicClick}
            disabled={isProcessing || allDone}
            id="mic-button"
            aria-label={isRecording ? 'Stop recording' : 'Start recording'}
          >
            {isProcessing ? (
              <span className="spinner"></span>
            ) : isRecording ? (
              '⏹️'
            ) : allDone ? (
              '✅'
            ) : (
              '🎙️'
            )}
          </button>
        </div>

        <p className="mic-hint">
          {isRecording
            ? 'Recording... Click to stop'
            : isProcessing
            ? 'Processing your voice...'
            : allDone
            ? 'All fields have been filled!'
            : `Click mic to fill: ${activeField?.labelEn || ''}`}
        </p>

        {/* Language Selector */}
        <div className="language-selector" id="language-selector">
          {LANGUAGES.map((lang) => (
            <button
              key={lang.code}
              className={`lang-chip${language === lang.code ? ' active' : ''}`}
              onClick={() => onLanguageChange(lang.code)}
              disabled={isRecording || isProcessing}
            >
              {lang.label}
            </button>
          ))}
        </div>

        {/* Transcript Display */}
        {transcript && (
          <div className="transcript-box fade-in" id="transcript-box">
            <div className="label">Transcript</div>
            <div className="text">{transcript}</div>
          </div>
        )}

        {/* Translated Text */}
        {translatedText && translatedText !== transcript && (
          <div className="transcript-box fade-in" style={{ marginTop: '8px' }}>
            <div className="label">English Translation</div>
            <div className="text">{translatedText}</div>
          </div>
        )}
      </div>

      {/* --- Progress Card --- */}
      <div className="progress-card" id="progress-card">
        <div className="progress-header">
          <h4>Form Completion</h4>
          <span className="progress-count">{fieldsFilled}/6 fields</span>
        </div>
        <div className="progress-bar-track">
          <div
            className="progress-bar-fill"
            style={{ width: `${progressPercent}%` }}
          ></div>
        </div>

        {/* Field checklist */}
        <div className="field-checklist">
          {FORM_FIELDS.map((field, index) => (
            <div
              key={field.key}
              className={`checklist-item${
                index === activeFieldIndex ? ' current' : ''
              }${index < activeFieldIndex || (fieldsFilled >= 6) ? ' done' : ''}`}
            >
              <span className="checklist-dot">
                {index < activeFieldIndex || fieldsFilled >= 6 ? '✓' : index === activeFieldIndex ? '●' : '○'}
              </span>
              <span className="checklist-label">{field.labelEn}</span>
            </div>
          ))}
        </div>
      </div>

      {/* --- Status Message --- */}
      {status && (
        <div className={`status-message ${status.type} fade-in`} id="status-message">
          {status.message}
        </div>
      )}
    </div>
  );
}
