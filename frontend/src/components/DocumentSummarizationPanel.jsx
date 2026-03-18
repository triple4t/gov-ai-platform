import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Upload, FileText, Loader2, CheckCircle, AlertCircle, RotateCcw } from 'lucide-react';

const ACCEPT = '.pdf,.doc,.docx,.png,.jpg,.jpeg,.tiff';
const API_SUMMARIZE = 'http://localhost:8000/api/v1/summarize';
const STORAGE_KEY = 'gov-ai-document-summary';

function loadPersistedSummary() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const data = JSON.parse(raw);
    if (data && data.result && (data.result.summary || (data.result.key_points && data.result.key_points.length))) {
      return data;
    }
  } catch (_) {}
  return null;
}

function savePersistedSummary(fileName, fileSize, result) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      fileName: fileName || null,
      fileSize: fileSize ?? null,
      result: { summary: result.summary || '', key_points: result.key_points || [], raw_text_preview: result.raw_text_preview || null },
    }));
  } catch (_) {}
}

export default function DocumentSummarizationPanel() {
  const [file, setFile] = useState(null);
  const [fileInfo, setFileInfo] = useState(null); // { name, size } when restored from localStorage
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    const persisted = loadPersistedSummary();
    if (persisted) {
      setFileInfo(persisted.fileName ? { name: persisted.fileName, size: persisted.fileSize } : null);
      setResult(persisted.result);
    }
  }, []);

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
    const dropped = e.dataTransfer.files[0];
    if (dropped) processFile(dropped);
  };

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) processFile(selected);
  };

  function processFile(selectedFile) {
    setFile(selectedFile);
    setFileInfo(null);
    setError('');
    setResult(null);
    handleSummarize(selectedFile);
  }

  const handleSummarize = async (fileToUse = null) => {
    const activeFile = fileToUse || file;
    if (!activeFile) {
      setError('Please select a document first.');
      return;
    }

    setIsLoading(true);
    setError('');
    setResult(null);

    const formData = new FormData();
    formData.append('file', activeFile);

    try {
      const response = await axios.post(API_SUMMARIZE, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      if (response.data.success) {
        const newResult = {
          summary: response.data.summary || '',
          key_points: response.data.key_points || [],
          raw_text_preview: response.data.raw_text_preview,
        };
        setResult(newResult);
        savePersistedSummary(activeFile.name, activeFile.size, newResult);
        setFileInfo({ name: activeFile.name, size: activeFile.size });
      } else {
        setError(response.data.message || 'Summarization failed.');
      }
    } catch (err) {
      const detail = err.response?.data?.detail;
      setError(Array.isArray(detail) ? detail.join(', ') : detail || err.message || 'Request failed.');
    } finally {
      setIsLoading(false);
    }
  };

  const clearAll = () => {
    setFile(null);
    setFileInfo(null);
    setResult(null);
    setError('');
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch (_) {}
  };

  const hasFile = file || fileInfo;
  const displayFileName = file?.name ?? fileInfo?.name;
  const displayFileSize = file?.size ?? fileInfo?.size;

  return (
    <div className="doc-summary-container">
      <div className="glass-panel doc-summary-panel doc-summary-upload-panel">
        <div className="panel-header-simple">
          <FileText size={18} className="icon-accent" />
          <h3>Document Summarization</h3>
        </div>
        <p className="doc-summary-intro">
          Generate concise summaries and structured insights from cabinet notes, tender documents, policy reports, and more. Supports PDF, Word (.doc/.docx), and images (including scanned documents).
        </p>

        <div
          className={`doc-summary-upload-area ${isDragging ? 'drag-active' : ''} ${hasFile ? 'has-file' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => document.getElementById('summary-file-input').click()}
        >
          <input
            type="file"
            id="summary-file-input"
            style={{ display: 'none' }}
            onChange={handleFileChange}
            accept={ACCEPT}
          />
          {hasFile ? (
            <div className="doc-summary-file-info">
              <CheckCircle size={24} className="file-done-icon" />
              <span className="file-name">{displayFileName}</span>
              <span className="file-size">{displayFileSize != null ? `${(displayFileSize / 1024).toFixed(1)} KB` : ''}</span>
            </div>
          ) : (
            <div className="doc-summary-upload-prompt">
              <Upload size={32} />
              <span>Click or drag document here</span>
              <span className="hint">PDF, DOC, DOCX, PNG, JPG, TIFF</span>
            </div>
          )}
        </div>

        {error && (
          <div className="doc-summary-error">
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}

        <div className="doc-summary-actions">
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => handleSummarize()}
            disabled={!file || isLoading}
            title={file ? undefined : 'Select a new document to summarize'}
          >
            {isLoading ? (
              <>
                <Loader2 className="spinner" size={18} />
                <span>Generating summary…</span>
              </>
            ) : (
              <>
                <FileText size={18} />
                <span>Summarize</span>
              </>
            )}
          </button>
          <button type="button" className="btn btn-secondary" onClick={clearAll}>
            <RotateCcw size={16} />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {result && (
        <div className="glass-panel doc-summary-result-panel">
          <div className="panel-header-simple">
            <FileText size={18} className="icon-accent" />
            <h3>Summary & key points</h3>
          </div>
          {result.summary && (
            <section className="doc-summary-result-section">
              <h4 className="doc-summary-result-heading">Summary</h4>
              <div className="doc-summary-result-text">
                {result.summary.split('\n').map((p, i) => (
                  <p key={i}>{p}</p>
                ))}
              </div>
            </section>
          )}
          {result.key_points && result.key_points.length > 0 && (
            <section className="doc-summary-result-section">
              <h4 className="doc-summary-result-heading">Key points</h4>
              <ul className="doc-summary-key-points">
                {result.key_points.map((point, i) => (
                  <li key={i}>{point}</li>
                ))}
              </ul>
            </section>
          )}
          {result.raw_text_preview && (
            <details className="doc-summary-debug">
              <summary>Extracted text preview</summary>
              <pre>{result.raw_text_preview}</pre>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
