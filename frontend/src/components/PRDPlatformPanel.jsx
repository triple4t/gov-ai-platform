import React, { useState, useCallback, useMemo } from 'react';
import axios from 'axios';
import { jsPDF } from 'jspdf';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  FileArchive,
  Loader2,
  Sparkles,
  ChevronRight,
  ChevronLeft,
  Download,
  FileText,
  ImageDown,
  RefreshCw,
  Code2,
  Layers,
  Upload,
  MessageSquare,
  Trash2,
  Plus,
} from 'lucide-react';
import { API_BASE } from '../config';
import { parsePrdPlain, formatSectionHeading } from '../utils/parsePrdPlain';
import { parseMarkdownForPdf } from '../utils/markdownToPdfStructure';

const API_PRD = `${API_BASE}/prd-platform`;
const API_CODE_SUM = `${API_BASE}/code-summarize`;
/** Hybrid RAG (BM25 + Chroma + LangGraph); same engine as /api/v1/rag — PRD-scoped alias */
const API_PRD_RAG = `${API_BASE}/prd-platform/rag`;

const RAG_FILE_ACCEPT =
  '.pdf,.doc,.docx,.txt,.md,.csv,.json,.html,.htm,.rtf,.png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff,.gif';

function makeRagSessionId() {
  return `doc_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 11)}`;
}

/** Capabilities whose output is long-form markdown: render as article + PDF. */
const MARKDOWN_DOC_CAPS = new Set(['tech_docs', 'architecture', 'code_review', 'sop']);

const techMarkdownComponents = {
  h1: (props) => {
    const { children, ...rest } = props;
    return (
      <h1 className="tech-md-h1" {...rest}>
        {children}
      </h1>
    );
  },
  h2: (props) => {
    const { children, ...rest } = props;
    return (
      <h2 className="tech-md-h2" {...rest}>
        {children}
      </h2>
    );
  },
  h3: (props) => {
    const { children, ...rest } = props;
    return (
      <h3 className="tech-md-h3" {...rest}>
        {children}
      </h3>
    );
  },
  p: (props) => {
    const { children, ...rest } = props;
    return (
      <p className="tech-md-p" {...rest}>
        {children}
      </p>
    );
  },
  ul: (props) => {
    const { children, ...rest } = props;
    return (
      <ul className="tech-md-ul" {...rest}>
        {children}
      </ul>
    );
  },
  ol: (props) => {
    const { children, ...rest } = props;
    return (
      <ol className="tech-md-ol" {...rest}>
        {children}
      </ol>
    );
  },
  li: (props) => {
    const { children, ...rest } = props;
    return (
      <li className="tech-md-li" {...rest}>
        {children}
      </li>
    );
  },
  blockquote: (props) => {
    const { children, ...rest } = props;
    return (
      <blockquote className="tech-md-blockquote" {...rest}>
        {children}
      </blockquote>
    );
  },
  a: (props) => {
    const { href, children, ...rest } = props;
    return (
      <a className="tech-md-a" href={href} target="_blank" rel="noopener noreferrer" {...rest}>
        {children}
      </a>
    );
  },
  code: (props) => {
    const { inline, children, className, ...rest } = props;
    if (inline) {
      return (
        <code className="tech-md-code-inline" {...rest}>
          {children}
        </code>
      );
    }
    return (
      <code className={className || 'tech-md-code-block'} {...rest}>
        {children}
      </code>
    );
  },
  pre: (props) => {
    const { children, ...rest } = props;
    return (
      <pre className="tech-md-pre" {...rest}>
        {children}
      </pre>
    );
  },
  table: (props) => {
    const { children, ...rest } = props;
    return (
      <div className="tech-md-table-wrap">
        <table className="tech-md-table" {...rest}>
          {children}
        </table>
      </div>
    );
  },
  th: (props) => {
    const { children, ...rest } = props;
    return (
      <th className="tech-md-th" {...rest}>
        {children}
      </th>
    );
  },
  td: (props) => {
    const { children, ...rest } = props;
    return (
      <td className="tech-md-td" {...rest}>
        {children}
      </td>
    );
  },
};

const SESSION_KEY = 'gov-ai-prd-project-id';

function getStoredProjectId() {
  try {
    return localStorage.getItem(SESSION_KEY) || '';
  } catch {
    return '';
  }
}

function setStoredProjectId(id) {
  try {
    if (id) localStorage.setItem(SESSION_KEY, id);
    else localStorage.removeItem(SESSION_KEY);
  } catch {
    /* localStorage unavailable */
  }
}

const STEPS = ['capability', 'questions', 'upload', 'generate', 'result'];

export default function PRDPlatformPanel() {
  /** prd = ZIP PRD platform flow; rag = hybrid RAG service (text ingest + query) */
  const [serviceTab, setServiceTab] = useState('prd');
  const [step, setStep] = useState('capability');
  const [capabilities, setCapabilities] = useState([]);
  const [capLoading, setCapLoading] = useState(true);
  const [capError, setCapError] = useState('');
  const [selectedCap, setSelectedCap] = useState(null);
  const [answers, setAnswers] = useState({});
  const [projectId, setProjectId] = useState(getStoredProjectId);
  const [uploadFile, setUploadFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState('');
  const [generating, setGenerating] = useState(false);
  const [genError, setGenError] = useState('');
  const [result, setResult] = useState(null);

  /** Code Summarizer: full indexed project summary only (POST /code-summarize/project). */
  const [csDetail, setCsDetail] = useState('medium');

  /** Hybrid RAG: one hidden session id for ingest + chat (no manual document_id field). */
  const ragDocumentId = useMemo(() => makeRagSessionId(), []);
  const [ragIndexed, setRagIndexed] = useState(false);
  const [ragLastFileName, setRagLastFileName] = useState('');
  const [ragQuestion, setRagQuestion] = useState('');
  const [ragBusy, setRagBusy] = useState(false);
  const [ragMsg, setRagMsg] = useState('');
  const [ragChat, setRagChat] = useState([]);
  const [ragDropActive, setRagDropActive] = useState(false);
  const ragChatEndRef = React.useRef(null);
  const ragFileInputRef = React.useRef(null);

  React.useEffect(() => {
    ragChatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [ragChat]);

  const resetCodeSummarizerForm = useCallback(() => {
    setCsDetail('medium');
  }, []);

  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const { data } = await axios.get(`${API_PRD}/capabilities`);
        if (!cancelled && data.capabilities) setCapabilities(data.capabilities);
      } catch (e) {
        if (!cancelled) setCapError(e.response?.data?.detail || e.message || 'Failed to load capabilities');
      } finally {
        if (!cancelled) setCapLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const currentQuestions = useMemo(() => {
    if (!selectedCap) return [];
    const c = capabilities.find((x) => x.id === selectedCap);
    return c?.clarifying_questions || [];
  }, [selectedCap, capabilities]);

  const needsUpload = useMemo(() => {
    if (!selectedCap) return false;
    const c = capabilities.find((x) => x.id === selectedCap);
    return Boolean(c?.requires_codebase);
  }, [selectedCap, capabilities]);

  const goCapability = useCallback(() => {
    setStep('capability');
    setGenError('');
  }, []);

  const selectCapability = useCallback((id) => {
    setSelectedCap(id);
    setAnswers({});
    setResult(null);
    setGenError('');
    resetCodeSummarizerForm();
    setStep('questions');
  }, [resetCodeSummarizerForm]);

  const goUpload = useCallback(() => {
    if (needsUpload) {
      setStep('upload');
      setUploadMsg('');
    } else {
      setStep('generate');
    }
  }, [needsUpload]);

  const handleAnswerChange = useCallback((key, value) => {
    setAnswers((prev) => ({ ...prev, [key]: value }));
  }, []);

  const validateQuestions = useCallback(() => {
    for (const q of currentQuestions) {
      if (q.required && !(answers[q.id] || '').trim()) return q.label;
    }
    return null;
  }, [currentQuestions, answers]);

  const handleNextFromQuestions = useCallback(() => {
    if (selectedCap === 'code_summarizer') {
      return;
    }
    const missing = validateQuestions();
    if (missing) {
      setGenError(`Please fill: ${missing}`);
      return;
    }
    setGenError('');
    goUpload();
  }, [validateQuestions, goUpload, selectedCap]);

  const handleUpload = useCallback(async () => {
    if (!uploadFile) {
      setUploadMsg('Choose a .zip file first.');
      return;
    }
    setUploading(true);
    setUploadMsg('');
    try {
      const fd = new FormData();
      fd.append('file', uploadFile);
      if (projectId) fd.append('project_id', projectId);
      const { data } = await axios.post(`${API_PRD}/projects/upload`, fd, {
        timeout: 600000,
      });
      const pid = data.project_id;
      setProjectId(pid);
      setStoredProjectId(pid);
      setUploadMsg(data.message || `Indexed ${data.chunk_count || 0} chunks.`);
      if (selectedCap === 'code_summarizer') {
        setStep('questions');
      } else {
        setStep('generate');
      }
    } catch (e) {
      setUploadMsg(e.response?.data?.detail || e.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  }, [uploadFile, projectId, selectedCap]);

  const handleCodeSummarizer = useCallback(async () => {
    setGenError('');
    setGenerating(true);
    setResult(null);
    const detail_level = csDetail;
    const pid = projectId.trim();
    if (!pid) {
      setGenError('Indexed project ID required. Upload a ZIP below or paste an existing project id.');
      setGenerating(false);
      return;
    }
    try {
      const { data } = await axios.post(
        `${API_CODE_SUM}/project`,
        { project_id: pid, detail_level },
        { timeout: 600_000 },
      );
      setResult({
        markdown: data.summary || '',
        message: `Detail level: ${data.level || detail_level}.`,
        content_format: 'code_summarizer',
        csMetadata: data.metadata,
      });
      setStep('result');
    } catch (e) {
      const detail = e.response?.data?.detail;
      setGenError(
        typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : e.message || 'Summarization failed',
      );
    } finally {
      setGenerating(false);
    }
  }, [csDetail, projectId]);

  const handleGenerate = useCallback(async () => {
    if (needsUpload && !projectId) {
      setGenError('Upload a project ZIP first (indexed project required).');
      return;
    }
    const missing = validateQuestions();
    if (missing) {
      setGenError(`Please fill: ${missing}`);
      return;
    }
    setGenerating(true);
    setGenError('');
    setResult(null);
    try {
      const { data } = await axios.post(`${API_PRD}/projects/generate`, {
        project_id: projectId || null,
        capability: selectedCap,
        answers,
      }, { timeout: 600000 });
      setResult(data);
      setStep('result');
    } catch (e) {
      setGenError(e.response?.data?.detail || e.message || 'Generation failed');
    } finally {
      setGenerating(false);
    }
  }, [needsUpload, projectId, selectedCap, answers, validateQuestions]);

  const isPrdPlain = useMemo(() => {
    if (!result?.markdown) return false;
    return selectedCap === 'prd' || result.content_format === 'prd_plain';
  }, [result, selectedCap]);

  const prdParsed = useMemo(() => {
    if (!isPrdPlain || !result?.markdown) return null;
    return parsePrdPlain(result.markdown);
  }, [isPrdPlain, result?.markdown]);

  const isCodeSummarizer = Boolean(
    selectedCap === 'code_summarizer' || result?.content_format === 'code_summarizer',
  );

  const isMarkdownDocCap = Boolean(
    selectedCap && MARKDOWN_DOC_CAPS.has(selectedCap) && result?.markdown,
  );

  const markdownPdfParsed = useMemo(() => {
    if (!isMarkdownDocCap || !result?.markdown) return null;
    return parseMarkdownForPdf(result.markdown);
  }, [isMarkdownDocCap, result?.markdown]);

  const canDownloadPdf = Boolean(prdParsed || markdownPdfParsed);

  const downloadText = useCallback(() => {
    if (!result?.markdown) return;
    const isPrd = selectedCap === 'prd' || result.content_format === 'prd_plain';
    const isCs = result.content_format === 'code_summarizer';
    const blob = new Blob([result.markdown], {
      type: isPrd ? 'text/plain;charset=utf-8' : 'text/markdown;charset=utf-8',
    });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    let name = `${selectedCap || 'output'}.md`;
    if (isPrd) name = `${selectedCap || 'prd'}.txt`;
    else if (isCs) name = 'full_project_code_summary.md';
    a.download = name;
    a.click();
    URL.revokeObjectURL(a.href);
  }, [result, selectedCap]);

  const downloadPdf = useCallback(() => {
    if (!result?.markdown) return;
    const doc = new jsPDF({ unit: 'pt', format: 'a4' });
    const margin = 48;
    const pageW = doc.internal.pageSize.getWidth();
    const maxW = pageW - margin * 2;
    let y = 56;
    const pageBottom = doc.internal.pageSize.getHeight() - 48;
    const ensureSpace = (h) => {
      if (y + h > pageBottom) {
        doc.addPage();
        y = 56;
      }
    };

    if (prdParsed) {
      const { title, sections } = prdParsed;
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(16);
      doc.splitTextToSize(title, maxW).forEach((line) => {
        ensureSpace(22);
        doc.text(line, margin, y);
        y += 22;
      });
      y += 10;
      doc.setFontSize(11);
      sections.forEach((sec) => {
        doc.setFont('helvetica', 'bold');
        const hLabel = formatSectionHeading(sec.heading);
        doc.splitTextToSize(hLabel, maxW).forEach((line) => {
          ensureSpace(16);
          doc.text(line, margin, y);
          y += 16;
        });
        y += 6;
        doc.setFont('helvetica', 'normal');
        sec.bullets.forEach((b) => {
          const text = typeof b === 'string' ? b : b.text;
          const depth = typeof b === 'string' ? 0 : b.depth || 0;
          const indent = margin + 14 + depth * 14;
          const wrapW = maxW - (14 + depth * 14);
          doc.splitTextToSize(`• ${text}`, wrapW).forEach((line) => {
            ensureSpace(14);
            doc.text(line, indent, y);
            y += 14;
          });
        });
        y += 12;
      });
    } else {
      const parsed = parseMarkdownForPdf(result.markdown);
      if (!parsed) return;
      const { title, parts } = parsed;
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(16);
      doc.splitTextToSize(title, maxW).forEach((line) => {
        ensureSpace(22);
        doc.text(line, margin, y);
        y += 22;
      });
      y += 10;
      doc.setFontSize(10);
      parts.forEach((part) => {
        if (part.type === 'h2') {
          doc.setFont('helvetica', 'bold');
          doc.setFontSize(12);
          doc.splitTextToSize(part.text, maxW).forEach((line) => {
            ensureSpace(15);
            doc.text(line, margin, y);
            y += 15;
          });
          y += 6;
          doc.setFontSize(10);
          doc.setFont('helvetica', 'normal');
        } else if (part.type === 'h3') {
          doc.setFont('helvetica', 'bold');
          doc.setFontSize(11);
          doc.splitTextToSize(part.text, maxW).forEach((line) => {
            ensureSpace(14);
            doc.text(line, margin, y);
            y += 14;
          });
          y += 4;
          doc.setFontSize(10);
          doc.setFont('helvetica', 'normal');
        } else if (part.type === 'bullet') {
          doc.setFont('helvetica', 'normal');
          const indent = margin + 12;
          doc.splitTextToSize(`• ${part.text}`, maxW - 12).forEach((line) => {
            ensureSpace(13);
            doc.text(line, indent, y);
            y += 13;
          });
        } else if (part.type === 'code') {
          doc.setFont('courier', 'normal');
          doc.setFontSize(8);
          const body = part.text.replace(/^```[\w.-]*\n?/, '').replace(/\n?```\s*$/, '').trim();
          doc.splitTextToSize(body, maxW).forEach((line) => {
            ensureSpace(11);
            doc.text(line, margin, y);
            y += 11;
          });
          doc.setFont('helvetica', 'normal');
          doc.setFontSize(10);
          y += 6;
        } else {
          doc.setFont('helvetica', 'normal');
          doc.splitTextToSize(part.text, maxW).forEach((line) => {
            ensureSpace(13);
            doc.text(line, margin, y);
            y += 13;
          });
          y += 4;
        }
      });
    }

    doc.save(`${selectedCap || 'document'}.pdf`);
  }, [prdParsed, result?.markdown, selectedCap]);

  const downloadDiagram = useCallback(() => {
    if (!result) return;
    const base = selectedCap || 'diagram';
    const trigger = (bytes, mime, ext) => {
      const blob = new Blob([bytes], { type: mime });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${base}.${ext}`;
      a.click();
      URL.revokeObjectURL(url);
    };
    if (result.svg_base64) {
      try {
        const bin = atob(result.svg_base64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
        trigger(bytes, 'image/svg+xml;charset=utf-8', 'svg');
      } catch {
        /* invalid base64 */
      }
      return;
    }
    if (result.png_base64) {
      try {
        const bin = atob(result.png_base64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
        trigger(bytes, 'image/png', 'png');
      } catch {
        /* invalid base64 */
      }
    }
  }, [result, selectedCap]);

  const hasDiagramDownload = Boolean(result?.svg_base64 || result?.png_base64);

  const resetFlow = useCallback(() => {
    setStep('capability');
    setSelectedCap(null);
    setAnswers({});
    setUploadFile(null);
    setUploadMsg('');
    setResult(null);
    setGenError('');
    resetCodeSummarizerForm();
  }, [resetCodeSummarizerForm]);

  const clearProject = useCallback(() => {
    setProjectId('');
    setStoredProjectId('');
    setUploadMsg('Cleared stored project. Upload a new ZIP to index.');
  }, []);

  const handleRagIngestFile = useCallback(
    async (file) => {
      if (!file) return;
      setRagBusy(true);
      setRagMsg('');
      setRagIndexed(false);
      try {
        const fd = new FormData();
        fd.append('file', file);
        fd.append('document_id', ragDocumentId);
        const { data } = await axios.post(`${API_PRD_RAG}/ingest-file`, fd, {
          timeout: 600000,
        });
        setRagChat([]);
        setRagIndexed(true);
        setRagLastFileName(file.name);
        setRagMsg(
          `Indexed ${data.chunks_indexed ?? 0} chunks from “${file.name}” in ${data.elapsed_seconds ?? '?'}s.`,
        );
      } catch (e) {
        const d = e.response?.data?.detail;
        setRagMsg(typeof d === 'string' ? d : d ? JSON.stringify(d) : e.message || 'File ingest failed');
        setRagIndexed(false);
        setRagLastFileName('');
      } finally {
        setRagBusy(false);
      }
    },
    [ragDocumentId],
  );

  const onRagFileInputChange = useCallback(
    (e) => {
      const f = e.target.files?.[0];
      e.target.value = '';
      if (f) handleRagIngestFile(f);
    },
    [handleRagIngestFile],
  );

  const handleRagQuery = useCallback(async () => {
    const q = ragQuestion.trim();
    if (!ragIndexed) {
      setRagMsg('Upload a document first, then ask a question.');
      return;
    }
    if (!q) {
      setRagMsg('Enter a question.');
      return;
    }
    const userId = Date.now();
    setRagChat((prev) => [...prev, { id: userId, role: 'user', content: q }]);
    setRagQuestion('');
    setRagBusy(true);
    setRagMsg('');
    try {
      const { data } = await axios.post(
        `${API_PRD_RAG}/query`,
        { document_id: ragDocumentId, question: q },
        { timeout: 600000 },
      );
      setRagChat((prev) => [
        ...prev,
        {
          id: userId + 1,
          role: 'assistant',
          content: data.answer || '_No answer text returned._',
          meta: {
            corrected_question: data.corrected_question,
            elapsed_seconds: data.elapsed_seconds,
            retries: data.retries,
          },
        },
      ]);
      setRagMsg(`Last reply in ${data.elapsed_seconds ?? '?'}s (retries=${data.retries ?? 0}).`);
    } catch (e) {
      const d = e.response?.data?.detail;
      const msg = typeof d === 'string' ? d : d ? JSON.stringify(d) : e.message || 'Query failed';
      setRagChat((prev) => [...prev, { id: userId + 1, role: 'error', content: msg }]);
      setRagMsg('');
    } finally {
      setRagBusy(false);
    }
  }, [ragDocumentId, ragIndexed, ragQuestion]);

  if (capLoading && serviceTab === 'prd') {
    return (
      <div className="glass-panel prd-platform-panel" style={{ padding: '48px', textAlign: 'center' }}>
        <Loader2 className="animate-spin" size={32} style={{ margin: '0 auto' }} />
        <p style={{ marginTop: 16 }}>Loading PRD Platform…</p>
      </div>
    );
  }

  if (capError && serviceTab === 'prd') {
    return (
      <div className="glass-panel prd-platform-panel" style={{ padding: '48px' }}>
        <p style={{ color: '#b91c1c' }}>{capError}</p>
      </div>
    );
  }

  const panelStyle =
    serviceTab === 'rag'
      ? { maxWidth: '100%', margin: '0 auto', padding: '8px 8px 12px' }
      : { maxWidth: '100%', margin: '0 auto', padding: '24px 20px' };

  return (
    <div className="prd-platform-panel" style={panelStyle}>
      <div className="glass-panel" style={{ padding: '24px', marginBottom: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
          <Sparkles size={28} className="text-accent" />
          <div>
            <h2 style={{ margin: 0, fontSize: '1.35rem' }}>AI-Based PRD Platform</h2>
            <p style={{ margin: '4px 0 0', opacity: 0.85, fontSize: '0.9rem' }}>
              PRD, technical docs, flow diagrams, SOPs, code review, architecture, dependency graphs, and full project code summary — with clarifying questions per capability.
            </p>
          </div>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginBottom: 12, alignItems: 'center' }}>
          <span style={{ fontSize: '0.8rem', opacity: 0.85, marginRight: 4 }}>Service:</span>
          <button
            type="button"
            className={serviceTab === 'prd' ? 'btn-primary' : 'btn-secondary'}
            style={{ padding: '8px 14px', fontSize: '0.85rem' }}
            onClick={() => setServiceTab('prd')}
          >
            PRD &amp; docs
          </button>
          <button
            type="button"
            className={serviceTab === 'rag' ? 'btn-primary' : 'btn-secondary'}
            style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '8px 14px', fontSize: '0.85rem' }}
            onClick={() => setServiceTab('rag')}
          >
            <Layers size={16} aria-hidden />
            Hybrid RAG
          </button>
        </div>
        {serviceTab === 'prd' ? (
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', fontSize: '0.8rem', opacity: 0.9 }}>
            {STEPS.map((s) => (
              <span
                key={s}
                style={{
                  padding: '4px 10px',
                  borderRadius: 999,
                  background: step === s ? 'var(--accent, #6366f1)' : 'rgba(0,0,0,0.06)',
                  color: step === s ? '#fff' : 'inherit',
                }}
              >
                {s}
              </span>
            ))}
          </div>
        ) : (
          <p style={{ margin: 0, fontSize: '0.85rem', opacity: 0.88 }}>
            BM25 + Chroma + Jina GGUF embeddings + LangGraph + local Qwen (same stack as <code>/api/v1/rag</code>). Index is{' '}
            <strong>in-memory</strong> on this API process — separate from PRD ZIP / FAISS.
          </p>
        )}
      </div>

      {serviceTab === 'rag' && (
        <div
          className="glass-panel rag-hybrid-chat"
          style={{
            minWidth: 0,
            padding: '18px',
            borderRadius: 14,
            border: '1px solid rgba(99, 102, 241, 0.18)',
            display: 'flex',
            flexDirection: 'column',
            minHeight: 'calc(100vh - 170px)',
            background: 'transparent',
          }}
        >
          <input
            ref={ragFileInputRef}
            type="file"
            accept={RAG_FILE_ACCEPT}
            style={{ display: 'none' }}
            onChange={onRagFileInputChange}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center', marginBottom: 10 }}>
            <button
              type="button"
              className="btn-secondary"
              disabled={ragBusy}
              onClick={() => ragFileInputRef.current?.click()}
              style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 12px', borderRadius: 999 }}
            >
              <span
                aria-hidden
                style={{
                  width: 22,
                  height: 22,
                  borderRadius: '50%',
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: 'rgba(99, 102, 241, 0.22)',
                }}
              >
                <Plus size={14} />
              </span>
              Upload document
            </button>
            {ragLastFileName ? (
              <span style={{ fontSize: '0.78rem', opacity: 0.85 }}>
                {ragIndexed ? 'Ready:' : 'File:'} <strong>{ragLastFileName}</strong>
              </span>
            ) : null}
          </div>
          <div
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                ragFileInputRef.current?.click();
              }
            }}
            onDragEnter={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setRagDropActive(true);
            }}
            onDragOver={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setRagDropActive(true);
            }}
            onDragLeave={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setRagDropActive(false);
            }}
            onDrop={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setRagDropActive(false);
              const f = e.dataTransfer.files?.[0];
              if (f) handleRagIngestFile(f);
            }}
            onClick={() => ragFileInputRef.current?.click()}
            style={{
              padding: '14px 14px',
              borderRadius: 10,
              border: `1.5px dashed ${ragDropActive ? 'rgba(129, 140, 248, 0.9)' : 'rgba(99, 102, 241, 0.5)'}`,
              background: ragDropActive ? 'rgba(99, 102, 241, 0.08)' : 'rgba(0,0,0,0.02)',
              textAlign: 'center',
              cursor: ragBusy ? 'not-allowed' : 'pointer',
              marginBottom: 14,
              opacity: ragBusy ? 0.7 : 1,
              fontSize: '0.82rem',
            }}
          >
            Drop PDF/DOCX/image here or click to upload
          </div>
          <div
            style={{
              flex: 1,
              overflowY: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: 12,
              padding: '12px 8px',
              marginBottom: 12,
              minHeight: 220,
              background: 'rgba(0,0,0,0.03)',
              borderRadius: 12,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, marginBottom: 12 }}>
              <h3 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: 8, fontSize: '1.12rem' }}>
                <MessageSquare size={22} aria-hidden /> Chat with the doc
              </h3>
              <button
                type="button"
                className="btn-secondary"
                disabled={ragChat.length === 0}
                onClick={() => setRagChat([])}
                title="Clear messages"
                style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 10px', fontSize: '0.8rem' }}
              >
                <Trash2 size={16} aria-hidden /> Clear
              </button>
            </div>
            <p style={{ fontSize: '0.82rem', opacity: 0.85, marginTop: 0, marginBottom: 12 }}>
              Ask follow-ups about the file you uploaded. <kbd style={{ fontSize: '0.75rem' }}>Enter</kbd> sends,{' '}
              <kbd style={{ fontSize: '0.75rem' }}>Shift+Enter</kbd> newline.
            </p>

            <div>
              {ragChat.length === 0 ? (
                <p style={{ margin: 'auto', textAlign: 'center', fontSize: '0.85rem', opacity: 0.65, maxWidth: 280 }}>
                  Ingest a document, then type a question here. Each reply stays in this thread.
                </p>
              ) : (
                ragChat.map((m) => {
                  if (m.role === 'user') {
                    return (
                      <div
                        key={m.id}
                        style={{
                          alignSelf: 'flex-end',
                          maxWidth: '92%',
                          padding: '10px 14px',
                          borderRadius: 12,
                          background: 'rgba(99, 102, 241, 0.18)',
                          border: '1px solid rgba(99, 102, 241, 0.25)',
                        }}
                      >
                        <div style={{ fontSize: '0.72rem', opacity: 0.75, marginBottom: 4 }}>You</div>
                        <div style={{ fontSize: '0.9rem', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{m.content}</div>
                      </div>
                    );
                  }
                  if (m.role === 'error') {
                    return (
                      <div
                        key={m.id}
                        style={{
                          alignSelf: 'stretch',
                          padding: '10px 12px',
                          borderRadius: 10,
                          background: 'rgba(185, 28, 28, 0.08)',
                          border: '1px solid rgba(185, 28, 28, 0.25)',
                          fontSize: '0.85rem',
                          color: '#991b1b',
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word',
                        }}
                      >
                        {m.content}
                      </div>
                    );
                  }
                  return (
                    <div
                      key={m.id}
                      style={{
                        alignSelf: 'flex-start',
                        width: '100%',
                        maxWidth: '100%',
                        padding: '12px 14px',
                        borderRadius: 12,
                        background: 'rgba(255,255,255,0.55)',
                        border: '1px solid rgba(0,0,0,0.06)',
                      }}
                    >
                      <div style={{ fontSize: '0.72rem', opacity: 0.7, marginBottom: 6 }}>Assistant</div>
                      <article className="tech-markdown-doc prd-document" style={{ fontSize: '0.88rem' }}>
                        <ReactMarkdown remarkPlugins={[remarkGfm]} components={techMarkdownComponents}>
                          {m.content}
                        </ReactMarkdown>
                      </article>
                      {m.meta?.corrected_question ? (
                        <p style={{ marginTop: 10, fontSize: '0.78rem', opacity: 0.82 }}>
                          <em>Retrieval query:</em> {m.meta.corrected_question}
                        </p>
                      ) : null}
                    </div>
                  );
                })
              )}
              <div ref={ragChatEndRef} />
            </div>
          </div>
          <label style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 8 }}>
            <span style={{ fontSize: '0.82rem', fontWeight: 600 }}>Your question</span>
            <textarea
              className="input-like"
              rows={3}
              value={ragQuestion}
              onChange={(e) => setRagQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  if (!ragBusy && ragIndexed) handleRagQuery();
                }
              }}
              placeholder={ragIndexed ? 'Ask about this document…' : 'Upload a document first'}
              disabled={!ragIndexed}
              style={{ fontFamily: 'inherit', resize: 'vertical', borderRadius: 10, opacity: ragIndexed ? 1 : 0.65 }}
            />
          </label>
          <button
            type="button"
            className="btn-primary"
            disabled={ragBusy || !ragIndexed || !ragQuestion.trim()}
            onClick={handleRagQuery}
          >
            {ragBusy ? <Loader2 className="animate-spin" size={18} /> : <Sparkles size={18} />}
            {ragBusy ? ' Querying…' : ' Send'}
          </button>
          {ragMsg ? (
            <pre
              style={{
                marginTop: 10,
                marginBottom: 0,
                padding: 10,
                background: 'rgba(0,0,0,0.04)',
                borderRadius: 8,
                fontSize: '0.78rem',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
              }}
            >
              {ragMsg}
            </pre>
          ) : null}
        </div>
      )}

      {serviceTab === 'prd' && step === 'capability' && (
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ marginTop: 0 }}>1. Choose capability</h3>
          <div style={{ display: 'grid', gap: 10 }}>
            {capabilities.map((c) => (
              <button
                key={c.id}
                type="button"
                onClick={() => selectCapability(c.id)}
                className="btn-primary"
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  textAlign: 'left',
                  padding: '14px 18px',
                  borderRadius: 12,
                  border: '1px solid rgba(99,102,241,0.35)',
                  background: 'rgba(99,102,241,0.08)',
                  cursor: 'pointer',
                  color: 'inherit',
                  fontSize: '0.95rem',
                }}
              >
                <span style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                  {c.id === 'code_summarizer' ? (
                    <Code2 size={22} style={{ flexShrink: 0, marginTop: 2, opacity: 0.9 }} aria-hidden />
                  ) : null}
                  <span>
                    <strong>{c.title}</strong>
                    <div style={{ fontSize: '0.82rem', opacity: 0.85, marginTop: 4 }}>{c.description}</div>
                  </span>
                </span>
                <ChevronRight size={20} />
              </button>
            ))}
          </div>
        </div>
      )}

      {serviceTab === 'prd' && step === 'questions' && selectedCap && selectedCap === 'code_summarizer' && (
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ marginTop: 0, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Code2 size={22} /> 2. Full project code summary
          </h3>
          <p style={{ fontSize: '0.9rem', opacity: 0.9, marginBottom: 12 }}>
            Summarizes your <strong>indexed</strong> repository (purpose, tech stack, architecture, key modules, flows). Upload a ZIP first or paste a{' '}
            <code>project_id</code> you already have.
          </p>
          <p
            style={{
              fontSize: '0.88rem',
              opacity: 0.88,
              marginBottom: 16,
              padding: '10px 12px',
              background: 'rgba(99,102,241,0.06)',
              borderRadius: 8,
              border: '1px solid rgba(99,102,241,0.15)',
            }}
          >
            No pasted source needed—the model uses RAG over chunks from your index.
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 16 }}>
            <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <span style={{ fontSize: '0.85rem' }}>Detail level</span>
              <select
                className="input-like"
                value={csDetail}
                onChange={(e) => setCsDetail(e.target.value)}
                style={{ padding: 8, borderRadius: 8, minWidth: 140 }}
              >
                <option value="short">Short</option>
                <option value="medium">Medium</option>
                <option value="detailed">Detailed</option>
              </select>
            </label>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <label style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <span>Indexed project ID</span>
              <input
                type="text"
                className="input-like"
                value={projectId}
                onChange={(e) => {
                  setProjectId(e.target.value);
                  setStoredProjectId(e.target.value);
                }}
                placeholder="UUID from Upload / index ZIP"
                style={{ padding: 10, borderRadius: 8, border: '1px solid rgba(0,0,0,0.12)' }}
              />
            </label>
          </div>

          {genError ? <p style={{ color: '#b91c1c', marginTop: 12 }}>{genError}</p> : null}
          <div style={{ display: 'flex', gap: 12, marginTop: 20, flexWrap: 'wrap', alignItems: 'center' }}>
            <button type="button" className="btn-secondary" onClick={goCapability} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <ChevronLeft size={18} /> Back
            </button>
            <button type="button" className="btn-secondary" onClick={() => setStep('upload')}>
              <FileArchive size={18} style={{ verticalAlign: 'middle', marginRight: 6 }} />
              Upload / index ZIP
            </button>
            <button type="button" className="btn-primary" disabled={generating} onClick={handleCodeSummarizer}>
              {generating ? <Loader2 className="animate-spin" size={18} /> : <Sparkles size={18} />}
              {generating ? ' Summarizing…' : ' Generate full summary'}
            </button>
          </div>
        </div>
      )}

      {serviceTab === 'prd' && step === 'questions' && selectedCap && selectedCap !== 'code_summarizer' && (
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ marginTop: 0 }}>2. Clarifying questions</h3>
          {currentQuestions.length === 0 ? (
            <p>No extra questions for this capability. Continue.</p>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {currentQuestions.map((q) => (
                <label key={q.id} style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  <span>
                    {q.label}
                    {q.required ? <span style={{ color: '#b91c1c' }}> *</span> : null}
                  </span>
                  {q.type === 'textarea' ? (
                    <textarea
                      className="input-like"
                      rows={4}
                      value={answers[q.id] || ''}
                      onChange={(e) => handleAnswerChange(q.id, e.target.value)}
                      placeholder={q.placeholder || ''}
                      style={{ width: '100%', padding: 10, borderRadius: 8, border: '1px solid rgba(0,0,0,0.12)' }}
                    />
                  ) : (
                    <input
                      type="text"
                      className="input-like"
                      value={answers[q.id] || ''}
                      onChange={(e) => handleAnswerChange(q.id, e.target.value)}
                      placeholder={q.placeholder || ''}
                      style={{ width: '100%', padding: 10, borderRadius: 8, border: '1px solid rgba(0,0,0,0.12)' }}
                    />
                  )}
                </label>
              ))}
            </div>
          )}
          {genError ? <p style={{ color: '#b91c1c', marginTop: 12 }}>{genError}</p> : null}
          <div style={{ display: 'flex', gap: 12, marginTop: 20 }}>
            <button type="button" className="btn-secondary" onClick={goCapability} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <ChevronLeft size={18} /> Back
            </button>
            <button type="button" className="btn-primary" onClick={handleNextFromQuestions}>
              Continue <ChevronRight size={18} style={{ verticalAlign: 'middle' }} />
            </button>
          </div>
        </div>
      )}

      {serviceTab === 'prd' && step === 'upload' && (
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ marginTop: 0 }}>3. Upload project (.zip)</h3>
          {selectedCap === 'code_summarizer' ? (
            <p style={{ fontSize: '0.9rem', opacity: 0.9, marginBottom: 8 }}>
              After indexing, you return to <strong>Full project code summary</strong> with <code>project_id</code> set.
            </p>
          ) : null}
          <p style={{ fontSize: '0.9rem', opacity: 0.9 }}>
            Code is chunked, embedded (local GGUF or Azure depending on server config), then indexed with FAISS. Large folders like <code>node_modules</code> are skipped. Indexing can take several minutes for large repos.
          </p>
          {projectId ? (
            <p style={{ fontSize: '0.85rem' }}>
              Current indexed project: <code>{projectId.slice(0, 8)}…</code>{' '}
              <button type="button" className="btn-secondary" style={{ marginLeft: 8 }} onClick={clearProject}>
                Clear
              </button>
            </p>
          ) : null}
          <label
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: 12,
              padding: 32,
              border: '2px dashed rgba(99,102,241,0.4)',
              borderRadius: 12,
              cursor: 'pointer',
              marginTop: 12,
            }}
          >
            <FileArchive size={40} />
            <span>Select ZIP archive</span>
            <input
              type="file"
              accept=".zip,application/zip"
              style={{ display: 'none' }}
              onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
            />
          </label>
          {uploadFile ? <p style={{ fontSize: '0.9rem' }}>Selected: {uploadFile.name}</p> : null}
          {uploadMsg ? <p style={{ marginTop: 8, fontSize: '0.9rem' }}>{uploadMsg}</p> : null}
          <div style={{ display: 'flex', gap: 12, marginTop: 20, flexWrap: 'wrap' }}>
            <button type="button" className="btn-secondary" onClick={() => setStep('questions')}>
              <ChevronLeft size={18} style={{ verticalAlign: 'middle' }} /> Back
            </button>
            <button type="button" className="btn-primary" disabled={uploading} onClick={handleUpload}>
              {uploading ? <Loader2 className="animate-spin" size={18} /> : null}
              {uploading ? ' Indexing…' : ' Upload & index'}
            </button>
          </div>
        </div>
      )}

      {serviceTab === 'prd' && step === 'generate' && selectedCap !== 'code_summarizer' && (
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ marginTop: 0 }}>4. Generate</h3>
          <p style={{ fontSize: '0.9rem' }}>
            Capability: <strong>{capabilities.find((c) => c.id === selectedCap)?.title}</strong>
          </p>
          {needsUpload ? (
            <p style={{ fontSize: '0.85rem' }}>
              Codebase: {projectId ? <code>{projectId.slice(0, 8)}…</code> : <span style={{ color: '#b91c1c' }}>not indexed</span>}
            </p>
          ) : null}
          {genError ? <p style={{ color: '#b91c1c' }}>{genError}</p> : null}
          <div style={{ display: 'flex', gap: 12, marginTop: 16, flexWrap: 'wrap' }}>
            {!needsUpload ? (
              <button type="button" className="btn-secondary" onClick={() => setStep('questions')}>
                Back
              </button>
            ) : (
              <button type="button" className="btn-secondary" onClick={() => setStep('upload')}>
                Back to upload
              </button>
            )}
            <button type="button" className="btn-primary" disabled={generating} onClick={handleGenerate}>
              {generating ? <Loader2 className="animate-spin" size={18} /> : <Sparkles size={18} />}
              {generating ? ' Generating…' : ' Generate'}
            </button>
          </div>
          {generating ? (
            <p style={{ marginTop: 12, fontSize: '0.85rem', opacity: 0.85, maxWidth: '42rem' }}>
              Local models can take several minutes for long documents. Keep this tab open—the request timeout is
              10 minutes. Backend logs show
              <code style={{ margin: '0 4px' }}>PRD platform generate started</code>
              then
              <code style={{ margin: '0 4px' }}>finished</code>
              when done.
            </p>
          ) : null}
        </div>
      )}

      {serviceTab === 'prd' && step === 'result' && result && (
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
            <h3 style={{ margin: 0 }}>5. Output</h3>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <button type="button" className="btn-secondary" onClick={downloadText} disabled={!result.markdown}>
                <Download size={18} style={{ verticalAlign: 'middle' }} />{' '}
                {isPrdPlain ? 'Download text' : 'Download Markdown'}
              </button>
              {canDownloadPdf ? (
                <button type="button" className="btn-secondary" onClick={downloadPdf}>
                  <FileText size={18} style={{ verticalAlign: 'middle' }} /> Download PDF
                </button>
              ) : null}
              {hasDiagramDownload ? (
                <button type="button" className="btn-secondary" onClick={downloadDiagram}>
                  <ImageDown size={18} style={{ verticalAlign: 'middle' }} />{' '}
                  {result.svg_base64 ? 'Download SVG' : 'Download PNG'}
                </button>
              ) : null}
              <button type="button" className="btn-secondary" onClick={resetFlow}>
                <RefreshCw size={18} style={{ verticalAlign: 'middle' }} /> New run
              </button>
            </div>
          </div>
          {result.message ? <p style={{ fontSize: '0.9rem', opacity: 0.9 }}>{result.message}</p> : null}
          {result.svg_base64 ? (
            <div style={{ marginTop: 16, border: '1px solid rgba(0,0,0,0.1)', borderRadius: 8, overflow: 'auto', background: '#fff' }}>
              <img
                src={`data:image/svg+xml;base64,${result.svg_base64}`}
                alt="Diagram"
                style={{ maxWidth: '100%', height: 'auto', display: 'block' }}
              />
            </div>
          ) : null}
          {result.png_base64 ? (
            <div style={{ marginTop: 16 }}>
              <img
                src={`data:image/png;base64,${result.png_base64}`}
                alt="Diagram"
                style={{ maxWidth: '100%', height: 'auto' }}
              />
            </div>
          ) : null}
          {prdParsed ? (
            <article className="prd-document">
              <h1 className="prd-document-title">{prdParsed.title}</h1>
              {prdParsed.sections.map((sec) => (
                <section key={sec.heading} className="prd-document-section">
                  <h2 className="prd-document-section-title">{formatSectionHeading(sec.heading)}</h2>
                  <ul className="prd-document-list">
                    {sec.bullets.map((b, idx) => {
                      const text = typeof b === 'string' ? b : b.text;
                      const depth = typeof b === 'string' ? 0 : b.depth || 0;
                      return (
                        <li
                          key={`${sec.heading}-${idx}`}
                          className={`prd-document-li${depth ? ' prd-document-li--nested' : ''}`}
                          style={depth ? { marginLeft: `${depth}rem` } : undefined}
                        >
                          {text}
                        </li>
                      );
                    })}
                  </ul>
                </section>
              ))}
            </article>
          ) : isCodeSummarizer ? (
            <article className="tech-markdown-doc prd-document" style={{ marginTop: 16 }}>
              <ReactMarkdown remarkPlugins={[remarkGfm]} components={techMarkdownComponents}>
                {result.markdown || '_No summary returned._'}
              </ReactMarkdown>
              {result.csMetadata && Object.keys(result.csMetadata).length > 0 ? (
                <>
                  <h3 className="prd-document-section-title" style={{ marginTop: 24 }}>
                    Metadata
                  </h3>
                  <pre
                    style={{
                      marginTop: 8,
                      padding: 14,
                      background: 'rgba(0,0,0,0.04)',
                      borderRadius: 8,
                      fontSize: '0.8rem',
                      overflow: 'auto',
                      maxHeight: 320,
                    }}
                  >
                    {JSON.stringify(result.csMetadata, null, 2)}
                  </pre>
                </>
              ) : null}
            </article>
          ) : isMarkdownDocCap ? (
            <article className="tech-markdown-doc prd-document">
              <ReactMarkdown remarkPlugins={[remarkGfm]} components={techMarkdownComponents}>
                {result.markdown || ''}
              </ReactMarkdown>
            </article>
          ) : (
            <pre
              style={{
                marginTop: 16,
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                fontSize: '0.88rem',
                lineHeight: 1.5,
                padding: 16,
                background: 'rgba(0,0,0,0.04)',
                borderRadius: 8,
                maxHeight: 480,
                overflow: 'auto',
              }}
            >
              {result.markdown || '(no text)'}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
