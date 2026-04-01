import React from 'react';

/**
 * Form field definitions with bilingual labels, prompts, and validation.
 * The order here defines the guided fill sequence.
 */
export const FORM_FIELDS = [
  {
    key: 'full_name',
    labelEn: 'Full Name',
    labelHi: 'पूरा नाम',
    placeholder: 'e.g., Rajesh Kumar',
    prompt: 'Please tell me your full name',
    promptHi: 'कृपया अपना पूरा नाम बताएं',
    type: 'text',
    icon: '👤',
  },
  {
    key: 'aadhaar_number',
    labelEn: 'Aadhaar Number',
    labelHi: 'आधार संख्या',
    placeholder: 'XXXX XXXX XXXX',
    prompt: 'Please tell me your 12-digit Aadhaar number',
    promptHi: 'कृपया अपना 12 अंकों का आधार नंबर बताएं',
    type: 'text',
    pattern: '^\\d{12}$',
    icon: '🪪',
  },
  {
    key: 'pan_card',
    labelEn: 'PAN Card',
    labelHi: 'पैन कार्ड',
    placeholder: 'ABCDE1234F',
    prompt: 'Please tell me your PAN card number',
    promptHi: 'कृपया अपना पैन कार्ड नंबर बताएं',
    type: 'text',
    pattern: '^[A-Z0-9]{5,15}$',
    icon: '💳',
  },
  {
    key: 'age',
    labelEn: 'Age / Date of Birth',
    labelHi: 'आयु / जन्म तिथि',
    placeholder: 'e.g., 32 or 15/08/1990',
    prompt: 'Please tell me your age or date of birth',
    promptHi: 'कृपया अपनी आयु या जन्म तिथि बताएं',
    type: 'text',
    icon: '🎂',
  },
  {
    key: 'address',
    labelEn: 'Address',
    labelHi: 'पता',
    placeholder: 'House no, Street, City',
    prompt: 'Please tell me your complete address',
    promptHi: 'कृपया अपना पूरा पता बताएं',
    type: 'text',
    icon: '🏠',
  },
  {
    key: 'phone_number',
    labelEn: 'Phone Number',
    labelHi: 'फोन नंबर',
    placeholder: 'e.g., 9876543210',
    prompt: 'Please tell me your 10-digit phone number',
    promptHi: 'कृपया अपना 10 अंकों का फोन नंबर बताएं',
    type: 'tel',
    pattern: '^[6-9]\\d{9}$',
    icon: '📱',
  },
];

/**
 * Validate a field value against its regex pattern.
 */
function validateField(field, value) {
  if (!value || !field.pattern) return null;
  const regex = new RegExp(field.pattern);
  return regex.test(value.replace(/[\s\-]/g, ''));
}

/**
 * FormPanel - Right side of the UI.
 * Displays the government form with bilingual labels.
 * The active field (being recorded) is highlighted with a glowing border.
 */
export default function FormPanel({
  formType,
  onFormTypeChange,
  formValues,
  onFormValueChange,
  onReset,
  onSubmit,
  activeFieldIndex,
  onFieldClick,
}) {
  const formTypes = [
    { value: 'aadhaar', label: 'Aadhaar Enrolment' },
    { value: 'pan', label: 'PAN Card Application' },
    { value: 'general', label: 'General Government Form' },
  ];

  return (
    <div className="form-panel" id="form-panel">
      {/* --- Page Title Bar --- */}
      <div className="page-title-bar" style={{ padding: 0, marginBottom: '16px' }}>
        <div className="page-title-left">
          <div className="page-title-icon">🎤</div>
          <div className="page-title-text">
            <h2>Voice-Based Form Filling</h2>
            <p>Speak naturally to auto-fill government forms</p>
          </div>
        </div>
        <div className="page-title-right">
          <div className="form-selector">
            <select
              value={formType}
              onChange={(e) => onFormTypeChange(e.target.value)}
              id="form-type-selector"
            >
              {formTypes.map((ft) => (
                <option key={ft.value} value={ft.value}>
                  {ft.label}
                </option>
              ))}
            </select>
          </div>
          <button className="reset-btn" onClick={onReset} id="reset-button">
            ↺ Reset
          </button>
        </div>
      </div>

      {/* --- Form Header --- */}
      <div className="form-panel-header">
        <h3>
          {formTypes.find((ft) => ft.value === formType)?.label || 'Government Form'}
        </h3>
        <p>Fields marked with * are mandatory</p>
      </div>

      {/* --- Form Fields --- */}
      <div className="form-fields" id="form-fields">
        {FORM_FIELDS.map((field, index) => {
          const value = formValues[field.key] || '';
          const validation = validateField(field, value);
          const isActive = index === activeFieldIndex;
          const isFilled = value && value.trim() !== '';

          // A field is valid if it has no pattern, or if its pattern matches
          const isValid = !value || !field.pattern || validation === true;

          return (
            <div
              className={`form-group${isActive ? ' active-field' : ''}${isFilled ? ' completed-field' : ''}${!isValid ? ' error-field' : ''}`}
              key={field.key}
              id={`group-${field.key}`}
              onClick={() => onFieldClick && onFieldClick(index)}
              style={{ cursor: 'pointer' }}
            >
              {/* Active field indicator arrow */}
              {isActive && (
                <div className="active-field-indicator">
                  <span className="arrow-icon">▶</span>
                  <span className="indicator-text">Speak now</span>
                </div>
              )}

              {/* Completed checkmark */}
              {isFilled && !isActive && (
                <div className="field-check">✓</div>
              )}

              <div className="form-label">
                <span className="field-icon">{field.icon}</span>
                <span className="en">{field.labelEn}</span>
                <span className="hi">({field.labelHi})</span>
              </div>
              <input
                className={`form-input${isFilled ? ' filled' : ''}${isActive ? ' active-input' : ''}`}
                type={field.type}
                placeholder={field.placeholder}
                value={value}
                onChange={(e) => onFormValueChange(field.key, e.target.value)}
                id={`field-${field.key}`}
                readOnly={isActive}
              />
              {isFilled && field.pattern && (
                <span
                  className={`validation-badge ${validation ? 'valid' : 'invalid'}`}
                >
                  {validation ? '✓ Valid format' : '⚠ Check format'}
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* --- Submit Actions --- */}
      <div className="form-actions">
        <button className="btn btn-secondary" onClick={onReset}>
          Clear All
        </button>
        <button
          className="btn btn-primary"
          onClick={onSubmit}
          id="submit-button"
          disabled={!FORM_FIELDS.every((f) => {
            const val = formValues[f.key];
            if (!val || val.trim() === '') return false; // Must be filled
            if (f.pattern && !validateField(f, val)) return false; // Must be valid
            return true;
          })}
        >
          Submit Form
        </button>
      </div>
    </div>
  );
}
