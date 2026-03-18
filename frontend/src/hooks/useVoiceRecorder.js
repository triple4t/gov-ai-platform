import { useState, useRef, useCallback } from 'react';

const API_BASE = 'http://localhost:8000/api/v1';

/**
 * Custom hook for voice recording and backend communication.
 * Supports both full-form extraction and single-field guided mode.
 */
export function useVoiceRecorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [status, setStatus] = useState({ type: 'info', message: 'Click microphone to start voice input' });
  const [error, setError] = useState(null);

  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const wsRef = useRef(null);
  const transcriptRef = useRef(''); // Tracks committed (final) text
  const latestTranscriptRef = useRef(''); // Tracks absolute latest (including interim)

  /**
   * Start recording audio from the microphone.
   * @param {string} language - BCP-47 language code to pass to Deepgram
   */
  const startRecording = useCallback(async (language = 'hi-IN') => {
    try {
      setError(null);
      setTranscript('');
      setTranslatedText('');
      transcriptRef.current = '';
      setStatus({ type: 'info', message: '🎙️ Requesting microphone access...' });

      // Open WebSocket to backend proxy
      const wsUrl = `ws://localhost:8000/api/v1/stream-stt?language=${language}`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'Results' && data.channel?.alternatives?.[0]) {
            const currentStr = data.channel.alternatives[0].transcript;
            if (currentStr.trim() === '') return;

            const fullText = (transcriptRef.current + ' ' + currentStr).trim();
            latestTranscriptRef.current = fullText;

            if (data.is_final) {
              transcriptRef.current = fullText;
              setTranscript(fullText);
            } else {
              // Show interim partial results in UI
              setTranscript(fullText);
            }
          }
        } catch (err) {
          console.error('WS parse error', err);
        }
      };

      ws.onopen = async () => {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            sampleRate: 16000,
            echoCancellation: true,
            noiseSuppression: true,
          },
        });

        streamRef.current = stream;

        const mediaRecorder = new MediaRecorder(stream, {
          mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
            ? 'audio/webm;codecs=opus'
            : 'audio/webm',
        });

        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0 && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(event.data);
          }
        };

        mediaRecorderRef.current = mediaRecorder;
        mediaRecorder.start(250);
        setIsRecording(true);
        setStatus({ type: 'info', message: '🎙️ Recording... Speak now, then click to stop' });
      };

      ws.onerror = (err) => {
        console.error('WS Connection Error', err);
        setError('Failed to connect to voice server');
        setStatus({ type: 'error', message: '❌ Voice server connection failed' });
      };
    } catch (err) {
      console.error('Microphone access error:', err);
      setError('Microphone access denied. Please allow microphone permissions.');
      setStatus({ type: 'error', message: '❌ Microphone access denied' });
    }
  }, []);

  /**
   * Stop recording and instantly extract the final text result.
   */
  const stopRecordingForField = useCallback(async (language, fieldName, fieldDescription) => {
    if (!mediaRecorderRef.current || mediaRecorderRef.current.state === 'inactive') return null;

    setIsRecording(false);
    setStatus({ type: 'processing', message: `⏳ Extracting ${fieldDescription}...` });

    return new Promise((resolve) => {
      mediaRecorderRef.current.onstop = async () => {
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
          streamRef.current = null;
        }

        // Close WebSocket intentionally
        if (wsRef.current) {
          // Give Deepgram 300ms to flush any final is_final messages over the socket
          setTimeout(() => {
            if (wsRef.current) {
               wsRef.current.close();
               wsRef.current = null;
            }
          }, 300);
        }

        setIsProcessing(true);

        // Fetch latest transcript from the ref.
        // latestTranscriptRef contains the most complete text (including last-second interim).
        setTimeout(async () => {
             const finalRecordedText = latestTranscriptRef.current;
             
             if (!finalRecordedText || finalRecordedText.trim() === '') {
                 setStatus({ type: 'error', message: '⚠️ Nothing was heard. Please try again.' });
                 setTranscript('');
                 setIsProcessing(false);
                 resolve(null);
                 return;
             }

             try {
               const formData = new FormData();
               formData.append('text', finalRecordedText);
               formData.append('language', language);
               formData.append('field_name', fieldName);
               formData.append('field_description', fieldDescription);

               const response = await fetch(`${API_BASE}/extract-single-text-field`, {
                 method: 'POST',
                 body: formData,
               });

               const result = await response.json();

               if (result.success && result.value) {
                 setStatus({
                   type: 'success',
                   message: `✅ ${fieldDescription}: "${result.value}"`,
                 });
                 resolve({
                   value: result.value,
                   transcript: finalRecordedText,
                   translated: result.translated_text,
                 });
               } else {
                 setStatus({
                   type: 'error',
                   message: result.error || `Could not extract ${fieldDescription}.`,
                 });
                 resolve(null);
               }
             } catch (err) {
               console.error('API call error:', err);
               setStatus({ type: 'error', message: '❌ Backend connection failed' });
               resolve(null);
             } finally {
               setIsProcessing(false);
             }
        }, 400); // 400ms flush padding for deepgram socket return
      };

      mediaRecorderRef.current.stop();
    });
  }, []);

  /**
   * Reset all state.
   */
  const reset = useCallback(() => {
    setTranscript('');
    setTranslatedText('');
    setError(null);
    setStatus({ type: 'info', message: 'Click microphone to start voice input' });
  }, []);

  return {
    isRecording,
    isProcessing,
    transcript,
    translatedText,
    status,
    error,
    startRecording,
    stopRecordingForField,
    reset,
    setStatus,
  };
}
