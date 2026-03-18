import React, { useState, useRef, useCallback } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import { 
  Upload, 
  Camera, 
  RefreshCcw, 
  CheckCircle2, 
  XCircle, 
  ShieldCheck, 
  UserCheck, 
  Loader2, 
  ArrowRight,
  ShieldAlert
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function FaceVerificationPanel() {
  const [docFile, setDocFile] = useState(null);
  const [docPreview, setDocPreview] = useState(null);
  const [capturedImage, setCapturedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  
  const webcamRef = useRef(null);
  const fileInputRef = useRef(null);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setDocFile(file);
      const reader = new FileReader();
      reader.onload = () => setDocPreview(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const capture = useCallback(() => {
    const imageSrc = webcamRef.current.getScreenshot();
    setCapturedImage(imageSrc);
  }, [webcamRef]);

  const verifyIdentity = async () => {
    if (!docFile) return;
    
    setLoading(true);
    setResult(null);
    
    // Capture the image if not already captured
    let currentCapture = capturedImage;
    if (!currentCapture) {
      currentCapture = webcamRef.current.getScreenshot();
      setCapturedImage(currentCapture);
    }

    try {
      // Convert base64 capture to Blob
      const captureBlob = await fetch(currentCapture).then(res => res.blob());
      const captureFile = new File([captureBlob], "capture.jpg", { type: "image/jpeg" });

      const formData = new FormData();
      formData.append('doc_image', docFile);
      formData.append('capture_image', captureFile);

      // Point to our unified backend
      const response = await axios.post('http://localhost:8000/api/v1/face-verify', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(response.data);
    } catch (error) {
      console.error("Verification error:", error);
      setResult({
        success: false,
        status: "error",
        error: error.response?.data?.error || "Connection to identity server failed"
      });
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setDocFile(null);
    setDocPreview(null);
    setCapturedImage(null);
    setResult(null);
  };

  return (
    <div className="face-container fade-in">
      <div className="face-layout-main">
        {/* Step 1: Reference ID Upload */}
        <section className={`glass-panel face-step-panel ${docFile ? 'completed' : ''}`}>
          <div className="face-step-header">
            <div className="step-number">1</div>
            <div className="step-info">
              <h3>
                <span className="en">Reference ID Photo</span>
                <span className="hi">पहचान पत्र फोटो</span>
              </h3>
              <p>Upload your Aadhaar or PAN card photo for reference</p>
            </div>
          </div>
          
          <div 
            className={`face-upload-zone ${docFile ? 'has-content' : ''}`}
            onClick={() => fileInputRef.current.click()}
          >
            <input 
              type="file" 
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept="image/*"
              hidden 
            />
            {!docPreview ? (
              <div className="upload-prompt">
                <div className="upload-icon-circle">
                  <Upload size={32} />
                </div>
                <p className="main-prompt">Click to upload document</p>
                <p className="sub-prompt">Supports Aadhaar, PAN, Passport</p>
              </div>
            ) : (
              <img src={docPreview} alt="Reference Preview" className="face-preview-img" />
            )}
            
            {docFile && (
              <div className="upload-badge success">
                <CheckCircle2 size={14} /> {docFile.name}
              </div>
            )}
          </div>
        </section>

        {/* Arrow for visual flow */}
        <div className="face-flow-arrow">
          <ArrowRight size={32} />
        </div>

        {/* Step 2: Live Selfie Capture */}
        <section className={`glass-panel face-step-panel ${capturedImage ? 'completed' : ''}`}>
          <div className="face-step-header">
            <div className="step-number">2</div>
            <div className="step-info">
              <h3>
                <span className="en">Live Face Capture</span>
                <span className="hi">लाइव चेहरा फोटो</span>
              </h3>
              <p>Position your face in the camera frame</p>
            </div>
          </div>
          
          <div className={`face-camera-zone ${capturedImage ? 'has-content' : ''}`}>
            {!capturedImage ? (
              <div className="webcam-wrapper">
                <Webcam
                  audio={false}
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  videoConstraints={{ facingMode: "user" }}
                  className="face-webcam"
                />
                <div className="camera-overlay">
                  <div className="face-guide-oval"></div>
                  <div className="camera-scanning-line"></div>
                </div>
              </div>
            ) : (
              <img src={capturedImage} alt="Capture" className="face-preview-img" />
            )}
            
            {capturedImage && (
              <div className="upload-badge info">
                <Camera size={14} /> Live Selfie Captured
              </div>
            )}
          </div>
          
          <div className="camera-controls">
            {!capturedImage ? (
              <button 
                className="btn btn-secondary capture-btn" 
                onClick={capture}
                disabled={loading}
              >
                <Camera size={18} />
                <span>Capture Photo</span>
              </button>
            ) : (
              <button className="btn btn-secondary capture-btn" onClick={() => setCapturedImage(null)}>
                <RefreshCcw size={18} />
                <span>Retake</span>
              </button>
            )}
          </div>
        </section>
      </div>

      {/* Verification Actions & Results */}
      <div className="face-actions-section">
        <div className="glass-panel verification-console">
          <div className="console-header">
            <ShieldCheck size={20} className="icon-accent" />
            <h3>Verification Engine Status</h3>
          </div>
          
          <div className="console-body">
            {!result ? (
              <div className="status-placeholder">
                <UserCheck size={40} className="pulse" />
                <p>Ensure both photos are clear before starting verification</p>
              </div>
            ) : (
              <motion.div 
                className="verification-result-display"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <div className={`result-status-card ${result.verification_result === "Verified" ? 'success' : 'failed'}`}>
                  <div className="status-main">
                    {result.verification_result === "Verified" ? (
                      <CheckCircle2 size={48} />
                    ) : (
                      <ShieldAlert size={48} />
                    )}
                    <div className="status-text">
                      <h4>{result.verification_result === "Verified" ? "Identity Match Confirmed" : "Identity Mismatch"}</h4>
                      <p>{result.verification_result === "Verified" ? "Citizen identity verified against government record." : "Verification failed. Check results below."}</p>
                    </div>
                  </div>
                </div>

                <div className="result-metrics">
                  <div className="metric-card">
                    <label>Match Similarity</label>
                    <div className="metric-value">
                      <div className="metric-bar-bg">
                        <motion.div 
                          className="metric-bar-fill"
                          initial={{ width: 0 }}
                          animate={{ width: `${result.match_score || 0}%` }}
                          transition={{ delay: 0.2, duration: 1 }}
                        ></motion.div>
                      </div>
                      <span>{result.match_score !== undefined ? `${result.match_score}%` : '--%'}</span>
                    </div>
                  </div>
                  
                  <div className="collapsible-stats">
                    <div className="stat-entry">
                      <span className="stat-label">Liveness Status:</span>
                      <span className={`stat-value ${result.liveness_status === "Live" ? 'text-success' : 'text-error'}`}>
                        {result.liveness_status || '--'}
                      </span>
                    </div>
                    <div className="stat-entry">
                      <span className="stat-label">Eyes Detected:</span>
                      <span className="stat-value">{result.liveness_details?.eyes_detected || 0} / 2 required</span>
                    </div>
                    {result.match_details?.distance && (
                      <div className="stat-entry">
                        <span className="stat-label">MSE Distance:</span>
                        <span className="stat-value">{result.match_details.distance.toFixed(2)}</span>
                      </div>
                    )}
                  </div>
                </div>

                {result.error && (
                  <div className="console-error">
                    <XCircle size={16} />
                    <span>{result.error}</span>
                  </div>
                )}
              </motion.div>
            )}
          </div>
          
          <div className="console-footer">
            <button 
              className="btn btn-secondary" 
              onClick={reset}
              disabled={loading}
            >
              <RefreshCcw size={16} />
              <span>Reset All</span>
            </button>
            <button 
              className="btn btn-primary verify-main-btn" 
              onClick={verifyIdentity}
              disabled={!docFile || loading}
            >
              {loading ? (
                <>
                  <Loader2 className="spinner" size={20} />
                  <span>Processing AI Models...</span>
                </>
              ) : (
                <>
                  <ShieldCheck size={20} />
                  <span>Confirm Verification</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
