import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE } from '../config';
import { 
  Upload, 
  FileText, 
  FileImage, 
  Image as ImageIcon, 
  Loader2, 
  CheckCircle, 
  AlertCircle,
  Hash,
  MapPin,
  Calendar,
  Phone,
  RotateCcw,
  Save,
  Trash2
} from 'lucide-react';

const OCR_STORAGE_KEY = 'gov-ai-document-ocr';

function loadPersistedOCR() {
  try {
    const raw = localStorage.getItem(OCR_STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch (_) {}
  return null;
}

function savePersistedOCR(fileName, extractedData, rawText, barcodes) {
  try {
    localStorage.setItem(OCR_STORAGE_KEY, JSON.stringify({
      fileName: fileName || null,
      extractedData: extractedData || {},
      rawText: rawText || '',
      barcodes: barcodes || [],
    }));
  } catch (_) {}
}

/** Merge top-level extracted_data with nested fields/other_fields for robust form binding */
function coalesceExtractedForm(data) {
  if (!data || typeof data !== 'object') {
    return { name: '', id_number: '', date_of_birth: '', address: '', phone_number: '' };
  }
  const f = data.fields && typeof data.fields === 'object' ? data.fields : {};
  const o = data.other_fields && typeof data.other_fields === 'object' ? data.other_fields : {};

  const firstNonEmpty = (...vals) => {
    for (const v of vals) {
      if (v != null && String(v).trim()) return String(v).trim();
    }
    return '';
  };

  const pick = (top, ...keys) => firstNonEmpty(top, ...keys.flatMap((k) => [f[k], o[k]]));

  let address = pick(data.address, 'address', 'work_location', 'office_location');
  if (!address) {
    const city = firstNonEmpty(f.city, o.city);
    const state = firstNonEmpty(f.state, o.state);
    if (city) address = state ? `${city}, ${state}` : city;
  }

  return {
    name: pick(data.name, 'name', 'full_name', 'holder_name', 'employee_name', 'applicant_name'),
    id_number: pick(
      data.id_number,
      'id_number',
      'aadhaar_number',
      'uid',
      'pan_number',
      'passport_number',
      'license_number'
    ),
    date_of_birth: pick(data.date_of_birth, 'date_of_birth', 'dob', 'birth_date'),
    phone_number: pick(
      data.phone_number,
      'phone_number',
      'mobile',
      'mobile_number',
      'contact_number'
    ),
    address,
  };
}

export default function OCRPanel() {
  const [file, setFile] = useState(null);
  const [fileInfo, setFileInfo] = useState(null); // { name } when restored from localStorage
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [isHighlighted, setIsHighlighted] = useState(false);

  const [extractedData, setExtractedData] = useState({
    name: '',
    id_number: '',
    date_of_birth: '',
    address: '',
    phone_number: ''
  });

  const [barcodes, setBarcodes] = useState([]);
  const [rawText, setRawText] = useState('');

  useEffect(() => {
    const persisted = loadPersistedOCR();
    if (persisted && (persisted.rawText || Object.values(persisted.extractedData || {}).some(Boolean))) {
      setFileInfo(persisted.fileName ? { name: persisted.fileName } : null);
      setExtractedData({
        name: persisted.extractedData?.name ?? '',
        id_number: persisted.extractedData?.id_number ?? '',
        date_of_birth: persisted.extractedData?.date_of_birth ?? '',
        address: persisted.extractedData?.address ?? '',
        phone_number: persisted.extractedData?.phone_number ?? '',
      });
      setRawText(persisted.rawText || '');
      setBarcodes(persisted.barcodes || []);
    }
  }, []);

  const processFile = (selectedFile) => {
    if (!selectedFile) return;
    setFile(selectedFile);
    setFileInfo(null);
    setError('');

    // Create preview for image or pdf
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);

    // Auto-trigger extraction
    handleExtractData(selectedFile);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    processFile(droppedFile);
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    processFile(selectedFile);
  };

  const handleExtractData = async (fileToProcess = null) => {
    const activeFile = fileToProcess || file;
    if (!activeFile) {
      setError('Please select a file first.');
      return;
    }

    setIsLoading(true);
    setError('');
    setIsHighlighted(false);

    const formData = new FormData();
    formData.append('file', activeFile);

    try {
      // Point to the unified backend endpoint
      const response = await axios.post(`${API_BASE}/document-ocr`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      if (response.data.success) {
        const data = response.data.extracted_data || {};
        const newExtracted = coalesceExtractedForm(data);
        const newRawText = response.data.raw_text || '';
        const newBarcodes = response.data.barcodes || [];

        setExtractedData(newExtracted);
        setRawText(newRawText);
        setBarcodes(newBarcodes);
        setFileInfo({ name: activeFile.name });
        savePersistedOCR(activeFile.name, newExtracted, newRawText, newBarcodes);

        // Trigger highlight animation
        setIsHighlighted(true);
        setTimeout(() => setIsHighlighted(false), 2000);
      } else {
        setError(response.data.message || 'Failed to process document');
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'An error occurred during extraction');
    } finally {
      setIsLoading(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setExtractedData(prev => ({ ...prev, [name]: value }));
  };

  const clearAll = () => {
    setFile(null);
    setFileInfo(null);
    setPreviewUrl(null);
    setExtractedData({
      name: '',
      id_number: '',
      date_of_birth: '',
      address: '',
      phone_number: ''
    });
    setRawText('');
    setBarcodes([]);
    setError('');
    try {
      localStorage.removeItem(OCR_STORAGE_KEY);
    } catch (_) {}
  };

  const hasFile = file || fileInfo;

  const handleSubmit = () => {
    alert('Document data confirmed and saved to system records.');
  };

  return (
    <div className="ocr-container fade-in">
      <div className="ocr-left-col">
        {/* Upload Section */}
        <section className="glass-panel ocr-upload-panel">
          <div className="panel-header-simple">
            <Upload size={18} className="icon-accent" />
            <h3>Document Upload</h3>
          </div>
          
          <div className="upload-wrapper">
            <div
              className={`ocr-upload-area ${isDragging ? 'drag-active' : ''} ${hasFile ? 'has-file' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => document.getElementById('file-input').click()}
            >
              <input
                type="file"
                id="file-input"
                style={{ display: 'none' }}
                onChange={handleFileChange}
                accept=".jpg,.jpeg,.png,.pdf,.tiff"
              />
              {hasFile ? (
                <div className="upload-success-content">
                  <CheckCircle className="upload-done-icon" />
                  <span className="file-name">{file?.name ?? fileInfo?.name}</span>
                  {file?.size != null && (
                    <span className="file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                  )}
                </div>
              ) : (
                <div className="upload-prompt-content">
                  <div className="upload-circle">
                    <Upload size={28} />
                  </div>
                  <span className="upload-main-text">Click or drag document to upload</span>
                  <span className="upload-sub-text">Aadhaar, PAN, Passport, PDF or Image</span>
                </div>
              )}
            </div>

            {error && (
              <div className="ocr-error-msg">
                <AlertCircle size={16} />
                <span>{error}</span>
              </div>
            )}

            <button
              className="btn btn-primary ocr-extract-btn"
              onClick={() => handleExtractData()}
              disabled={!file || isLoading}
              title={file ? undefined : 'Upload a document to extract'}
            >
              {isLoading ? (
                <>
                  <Loader2 className="spinner" />
                  <span>Processing AI...</span>
                </>
              ) : (
                <>
                  <FileText size={18} />
                  <span>Extract Information</span>
                </>
              )}
            </button>
          </div>
        </section>

        {/* Preview Section */}
        <section className="glass-panel ocr-preview-panel">
          <div className="panel-header-simple">
            <ImageIcon size={18} className="icon-accent" />
            <h3>Document Preview</h3>
          </div>
          <div className="ocr-preview-content">
            {previewUrl ? (
              file?.type === "application/pdf" ? (
                <iframe
                  src={previewUrl}
                  className="ocr-pdf-iframe"
                  title="PDF Preview"
                />
              ) : (
                <img src={previewUrl} alt="Document Preview" className="ocr-preview-img" />
              )
            ) : (
              <div className="ocr-preview-placeholder">
                <FileImage size={40} />
                <p>Upload a document to see preview</p>
              </div>
            )}
          </div>
        </section>
      </div>

      <div className="ocr-right-col">
        {/* Results Section */}
        <section className="glass-panel ocr-results-panel">
          <div className="panel-header-simple">
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flex: 1 }}>
              <FileText size={18} className="icon-accent" />
              <h3>Extracted Form Fields</h3>
            </div>
            <button className="reset-btn" onClick={clearAll} title="Clear all fields">
              <RotateCcw size={14} />
              <span>Reset</span>
            </button>
          </div>
          
          <div className="ocr-form-content">
            <div className="ocr-form-group">
              <label>
                <FileText size={14} /> 
                <span className="en">Full Name</span>
                <span className="hi">नाम</span>
              </label>
              <input
                type="text"
                name="name"
                className={`form-input ${isHighlighted ? 'field-highlight' : ''}`}
                value={extractedData.name}
                onChange={handleInputChange}
                placeholder="Rajesh Kumar"
              />
            </div>

            <div className="ocr-form-group">
              <label>
                <Hash size={14} /> 
                <span className="en">ID Number</span>
                <span className="hi">पहचान संख्या</span>
              </label>
              <input
                type="text"
                name="id_number"
                className={`form-input ${isHighlighted ? 'field-highlight' : ''}`}
                value={extractedData.id_number}
                onChange={handleInputChange}
                placeholder="XXXX XXXX XXXX"
              />
            </div>

            <div className="ocr-form-row">
              <div className="ocr-form-group half">
                <label>
                  <Calendar size={14} /> 
                  <span className="en">Date of Birth</span>
                  <span className="hi">जन्म तिथि</span>
                </label>
                <input
                  type="text"
                  name="date_of_birth"
                  className={`form-input ${isHighlighted ? 'field-highlight' : ''}`}
                  value={extractedData.date_of_birth}
                  onChange={handleInputChange}
                  placeholder="DD/MM/YYYY"
                />
              </div>
              <div className="ocr-form-group half">
                <label>
                  <Phone size={14} /> 
                  <span className="en">Phone Number</span>
                  <span className="hi">फोन नंबर</span>
                </label>
                <input
                  type="text"
                  name="phone_number"
                  className={`form-input ${isHighlighted ? 'field-highlight' : ''}`}
                  value={extractedData.phone_number}
                  onChange={handleInputChange}
                  placeholder="+91 XXXXX XXXXX"
                />
              </div>
            </div>

            <div className="ocr-form-group">
              <label>
                <MapPin size={14} /> 
                <span className="en">Address</span>
                <span className="hi">पता</span>
              </label>
              <textarea
                name="address"
                className={`form-input ocr-textarea ${isHighlighted ? 'field-highlight' : ''}`}
                value={extractedData.address}
                onChange={handleInputChange}
                placeholder="House no, Street, City, State, ZIP"
                rows={3}
              />
            </div>

            {barcodes.length > 0 && (
              <div className="ocr-barcode-section">
                <label className="section-divider-label">Detected Barcodes / QR</label>
                <div className="barcode-list">
                  {barcodes.map((b, i) => (
                    <div key={i} className="barcode-item">
                      <span className="barcode-type">{b.type}</span>
                      <code className="barcode-data">{b.data}</code>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {rawText && (
              <div className="ocr-raw-text-section">
                <label className="section-divider-label">Raw OCR Text (Debug)</label>
                <div className="ocr-raw-display">
                  {rawText}
                </div>
              </div>
            )}

            <div className="form-actions" style={{ marginTop: 'auto', paddingTop: '24px', display: 'flex', gap: '16px', justifyContent: 'flex-end' }}>
              <button className="btn btn-secondary" onClick={clearAll} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Trash2 size={16} />
                <span>Discard</span>
              </button>
              <button 
                className="btn btn-primary" 
                onClick={handleSubmit}
                disabled={!extractedData.name && !extractedData.id_number}
                style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
              >
                <Save size={16} />
                <span>Confirm & Submit</span>
              </button>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
