import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { MessageCircle, Send, Loader2, User, Bot, Mic, Square, Volume2, Trash2 } from 'lucide-react';
import { API_BASE } from '../config';
const SESSION_STORAGE_KEY = 'gov-ai-chat-session-id';

const LANGUAGES = [
  { code: 'hi', label: 'हिंदी' },
  { code: 'en', label: 'English' },
  { code: 'mr', label: 'मराठी' },
  { code: 'bn', label: 'বাংলা' },
  { code: 'te', label: 'తెలుగు' },
  { code: 'ta', label: 'தமிழ்' },
];

function getOrCreateSessionId() {
  try {
    const stored = localStorage.getItem(SESSION_STORAGE_KEY);
    if (stored) return stored;
  } catch (_) {}
  const id = 'sess_' + Math.random().toString(36).slice(2) + Date.now().toString(36);
  try {
    localStorage.setItem(SESSION_STORAGE_KEY, id);
  } catch (_) {}
  return id;
}

export default function ConversationalAIPanel() {
  const [sessionId, setSessionId] = useState(getOrCreateSessionId);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [language, setLanguage] = useState('hi');
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [voiceError, setVoiceError] = useState('');
  const [playingTtsId, setPlayingTtsId] = useState(null);
  const messagesEndRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioRef = useRef(null);

  const scrollToBottom = () => messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  useEffect(() => scrollToBottom(), [messages]);

  const handleClear = () => {
    setMessages([]);
    // Setup a new session ID so the backend doesn't remember the previous dialogue block
    const newId = 'sess_' + Math.random().toString(36).slice(2) + Date.now().toString(36);
    try {
      localStorage.setItem(SESSION_STORAGE_KEY, newId);
    } catch (_) {}
    setSessionId(newId);
  };

  // Load persistent conversation history for this session
  useEffect(() => {
    if (!sessionId || historyLoaded) return;
    let cancelled = false;
    axios
      .get(`${API_BASE}/chat/history`, { params: { session_id: sessionId } })
      .then(({ data }) => {
        if (cancelled) return;
        if (data.messages && data.messages.length) {
          setMessages(data.messages);
          if (data.language) setLanguage(data.language);
        }
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setHistoryLoaded(true);
      });
    return () => { cancelled = true; };
  }, [sessionId, historyLoaded]);

  const startRecording = async () => {
    setVoiceError('');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks = [];
      recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunks, { type: 'audio/webm' });
        const form = new FormData();
        form.append('file', blob, 'recording.webm');
        form.append('language', language);
        setIsTranscribing(true);
        try {
          const { data } = await axios.post(`${API_BASE}/voice/speech-to-text`, form, {
            headers: { 'Content-Type': 'multipart/form-data' },
          });
          setInput((prev) => (prev ? `${prev} ${data.text}` : data.text).trim());
        } catch (err) {
          setVoiceError(err.response?.data?.detail || err.message || 'Voice recognition failed');
        } finally {
          setIsTranscribing(false);
        }
      };
      recorder.start();
      mediaRecorderRef.current = { recorder, stream };
      setIsRecording(true);
    } catch (err) {
      setVoiceError('Microphone access denied or unavailable');
    }
  };

  const stopRecording = () => {
    if (!mediaRecorderRef.current?.recorder || mediaRecorderRef.current.recorder.state === 'inactive') return;
    mediaRecorderRef.current.recorder.stop();
    mediaRecorderRef.current = null;
    setIsRecording(false);
  };

  const playTts = async (text, lang, msgId) => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    if (playingTtsId === msgId) {
      setPlayingTtsId(null);
      return;
    }
    setVoiceError('');
    setPlayingTtsId(msgId);
    try {
      const form = new FormData();
      form.append('text', text.slice(0, 500));
      form.append('language', lang);
      const { data } = await axios.post(`${API_BASE}/voice/text-to-speech`, form, {
        responseType: 'blob',
      });
      const url = URL.createObjectURL(data);
      const audio = new Audio(url);
      audioRef.current = audio;
      audio.onended = () => {
        URL.revokeObjectURL(url);
        setPlayingTtsId(null);
        audioRef.current = null;
      };
      audio.onerror = () => {
        setVoiceError('Playback failed');
        setPlayingTtsId(null);
      };
      await audio.play();
    } catch (err) {
      setVoiceError(err.response?.data?.detail || err.message || 'Speech playback failed');
      setPlayingTtsId(null);
    }
  };

  const handleSend = async (e) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || chatLoading) return;
    setInput('');
    setChatError('');
    setVoiceError('');
    const userMsg = { role: 'user', content: text };
    setMessages((prev) => [...prev, userMsg]);
    setChatLoading(true);
    try {
      const { data } = await axios.post(
        `${API_BASE}/chat/send`,
        { message: text, language, session_id: sessionId },
        { headers: { 'Content-Type': 'application/json' } }
      );
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: data.response || '' },
      ]);
    } catch (err) {
      setChatError(err.response?.data?.detail || err.message || 'Send failed');
      setMessages((prev) => prev.filter((m) => m !== userMsg));
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div className="conv-ai-container">
      <div className="conv-ai-chat-panel">
        <div className="conv-ai-chat-header">
          <div className="conv-ai-header-left">
            <MessageCircle size={22} className="conv-ai-logo" />
            <h3 className="conv-ai-title">Conversational AI</h3>
          </div>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <button 
              onClick={handleClear}
              className="conv-ai-lang-select" 
              style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', color: 'var(--color-error)' }}
              title="Clear Chat History"
            >
              <Trash2 size={16} /> <span>Clear</span>
            </button>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="conv-ai-lang-select"
              aria-label="Language"
            >
              {LANGUAGES.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="conv-ai-messages" ref={messagesEndRef}>
          {messages.length === 0 && (
            <div className="conv-ai-welcome">
              <MessageCircle size={48} className="conv-ai-welcome-icon" />
              <p>Ask about government services in your language.</p>
              <p className="conv-ai-welcome-hint">Try: &quot;How do I get an Aadhaar card?&quot; or &quot;आधार कार्ड कैसे बनवाएं?&quot;</p>
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`conv-ai-msg conv-ai-msg-${m.role}`}>
              {m.role === 'user' ? (
                <>
                  <div className="conv-ai-msg-bubble conv-ai-msg-bubble-user">{m.content}</div>
                  <div className="conv-ai-avatar conv-ai-avatar-user" aria-hidden>
                    <User size={18} />
                  </div>
                </>
              ) : (
                <>
                  <div className="conv-ai-avatar conv-ai-avatar-assistant" aria-hidden>
                    <Bot size={18} />
                  </div>
                  <div className="conv-ai-msg-bubble conv-ai-msg-bubble-assistant">
                    <span className="conv-ai-msg-text" style={{ whiteSpace: 'pre-wrap', lineHeight: '1.6', display: 'block' }}>{m.content}</span>
                    <button
                      type="button"
                      className="conv-ai-tts-btn"
                      onClick={() => playTts(m.content, language, i)}
                      title="Listen"
                      aria-label="Play speech"
                    >
                      {playingTtsId === i ? (
                        <Loader2 size={16} className="spinner" />
                      ) : (
                        <Volume2 size={16} />
                      )}
                    </button>
                  </div>
                </>
              )}
            </div>
          ))}
          {chatLoading && (
            <div className="conv-ai-msg conv-ai-msg-assistant">
              <div className="conv-ai-avatar conv-ai-avatar-assistant" aria-hidden>
                <Bot size={18} />
              </div>
              <div className="conv-ai-msg-bubble conv-ai-msg-bubble-assistant conv-ai-typing">
                <Loader2 className="spinner" size={18} /> Thinking...
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {(chatError || voiceError) && (
          <div className="conv-ai-error conv-ai-error-inline">{chatError || voiceError}</div>
        )}

        <div className="conv-ai-input-wrap">
          <form onSubmit={handleSend} className="conv-ai-send-form">
            <button
              type="button"
              className={`conv-ai-mic-btn ${isRecording ? 'recording' : ''}`}
              onClick={isRecording ? stopRecording : startRecording}
              disabled={chatLoading || isTranscribing}
              title={isRecording ? 'Stop recording' : 'Voice input'}
              aria-label={isRecording ? 'Stop recording' : 'Voice input'}
            >
              {isTranscribing ? (
                <Loader2 size={20} className="spinner" />
              ) : isRecording ? (
                <Square size={20} fill="currentColor" />
              ) : (
                <Mic size={20} />
              )}
            </button>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask anything... or use voice"
              className="conv-ai-input"
              disabled={chatLoading}
            />
            <button type="submit" className="conv-ai-send-btn" disabled={chatLoading || !input.trim()} title="Send">
              <Send size={20} />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
