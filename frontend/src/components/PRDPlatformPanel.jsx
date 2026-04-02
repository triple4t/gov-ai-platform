import React, { useState, useCallback, useMemo } from 'react';
import axios from 'axios';
import { jsPDF } from 'jspdf';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  FileArchive,
  Loader2,
  ChevronRight,
  ChevronLeft,
  Download,
  FileText,
  ImageDown,
  Code2,
  Search,
  Upload,
  MessageSquare,
  Trash2,
  Send,
  Plus,
  FolderArchive,
  Zap,
  CheckCircle2,
  RotateCcw,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
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
      <h1 className="tech-md-h1 font-display" {...rest}>
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

const STEP_LABELS = {
  capability: 'Capability',
  questions: 'Questions',
  upload: 'Upload',
  generate: 'Generate',
  result: 'Result',
};

const inputFieldClass =
  'w-full rounded-xl border border-input bg-card px-4 py-2.5 text-sm text-foreground shadow-sm transition-colors placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20';

const textareaFieldClass = `${inputFieldClass} resize-none`;

function StepIndicator({ current, steps }) {
  const currentIdx = steps.findIndex((s) => s === current);
  return (
    <div className="mb-6 flex items-center gap-1.5 overflow-x-auto pb-1" aria-label="Workflow steps">
      {steps.map((s, i) => (
        <div key={s} className="flex items-center gap-1.5">
          <span
            className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-all duration-300 ${
              i === currentIdx
                ? 'bg-primary text-primary-foreground shadow-sm'
                : i < currentIdx
                  ? 'bg-primary/15 text-primary'
                  : 'bg-secondary text-muted-foreground'
            }`}
          >
            <span
              className={`flex h-4 w-4 items-center justify-center rounded-full text-[10px] font-bold ${
                i === currentIdx
                  ? 'bg-primary-foreground/20 text-primary-foreground'
                  : i < currentIdx
                    ? 'bg-primary/20 text-primary'
                    : 'bg-muted-foreground/20 text-muted-foreground'
              }`}
            >
              {i < currentIdx ? '✓' : i + 1}
            </span>
            {STEP_LABELS[s] || s}
          </span>
          {i < steps.length - 1 && (
            <ChevronRight
              className={`h-3 w-3 shrink-0 ${i < currentIdx ? 'text-primary/40' : 'text-muted-foreground/30'}`}
            />
          )}
        </div>
      ))}
    </div>
  );
}

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
  const zipFileInputRef = React.useRef(null);

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
      <div className="flex min-h-[240px] flex-col items-center justify-center gap-4 rounded-2xl border border-border/50 bg-card/50 p-10">
        <Loader2 className="h-8 w-8 animate-spin text-primary" aria-hidden />
        <p className="text-sm text-muted-foreground">Loading DocuAtlas…</p>
      </div>
    );
  }

  if (capError && serviceTab === 'prd') {
    return (
      <div className="rounded-2xl border border-destructive/40 bg-destructive/5 p-6 text-sm text-destructive">
        <p>{capError}</p>
      </div>
    );
  }

  const ragFullScreen = serviceTab === 'rag';

  return (
    <div
      className={
        ragFullScreen
          ? 'fixed inset-x-0 bottom-0 top-16 z-40 flex flex-col bg-background'
          : 'flex min-h-0 flex-1 flex-col gap-4'
      }
    >
      <div
        className={`flex shrink-0 flex-wrap items-center gap-2 ${ragFullScreen ? 'border-b border-border/40 bg-card/80 px-4 py-3 backdrop-blur-xl' : ''}`}
      >
        <Button variant={serviceTab === 'prd' ? 'chip-active' : 'chip'} size="chip" onClick={() => setServiceTab('prd')}>
          <FileText className="h-3.5 w-3.5" fill="currentColor" />
          PRD &amp; Docs
        </Button>
        <Button variant={serviceTab === 'rag' ? 'chip-active' : 'chip'} size="chip" onClick={() => setServiceTab('rag')}>
          <Search className="h-3.5 w-3.5" fill="currentColor" />
          Hybrid RAG
        </Button>
        {ragFullScreen ? (
          <>
            <div className="mx-2 hidden h-6 w-px bg-border sm:block" aria-hidden />
            <span className="font-display text-sm font-semibold text-foreground">Document chat</span>
            <div className="flex-1" />
            <Button
              variant="ghost"
              size="sm"
              className="text-muted-foreground"
              onClick={() => {
                setRagChat([]);
                setRagMsg('');
                setRagLastFileName('');
                setRagIndexed(false);
              }}
            >
              <Trash2 className="h-4 w-4" /> New chat
            </Button>
          </>
        ) : null}
      </div>

      <div
        className={
          ragFullScreen
            ? 'flex min-h-0 flex-1 flex-col'
            : 'glass-panel flex min-h-0 flex-1 flex-col rounded-xl border border-border/50 p-4 sm:p-6 lg:p-8'
        }
      >
        {serviceTab === 'rag' ? (
          <div className="animate-fade-in flex min-h-0 flex-1 flex-col">
            <input
              ref={ragFileInputRef}
              type="file"
              accept={RAG_FILE_ACCEPT}
              className="hidden"
              onChange={onRagFileInputChange}
            />

            <div
              className={`relative flex min-h-0 flex-1 flex-col overflow-y-auto bg-background ${ragDropActive ? 'bg-primary/5' : ''}`}
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
                if (!e.currentTarget.contains(e.relatedTarget)) setRagDropActive(false);
              }}
              onDrop={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setRagDropActive(false);
                const f = e.dataTransfer.files?.[0];
                if (f) handleRagIngestFile(f);
              }}
            >
              {ragDropActive ? (
                <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center rounded-lg border-2 border-dashed border-primary/50 bg-primary/10">
                  <p className="text-sm font-medium text-primary">Drop file to index</p>
                </div>
              ) : null}

              <div className="mx-auto flex min-h-full w-full max-w-4xl flex-1 flex-col px-3 py-4 lg:max-w-5xl xl:max-w-6xl xl:px-8">
                {ragChat.length === 0 ? (
                  <div className="flex flex-1 flex-col items-center justify-center pb-32 text-center">
                    <MessageSquare className="mb-3 h-10 w-10 text-muted-foreground/35" strokeWidth={1.25} aria-hidden />
                    <p className="max-w-sm text-sm text-muted-foreground">
                      {ragIndexed ? 'Ask a question below.' : 'Attach a document with + to start.'}
                    </p>
                  </div>
                ) : (
                  <div className="space-y-6 pb-8">
                    {ragChat.map((m) => {
                      if (m.role === 'user') {
                        return (
                          <div key={m.id} className="animate-fade-in flex justify-end">
                            <div className="max-w-[min(100%,36rem)] rounded-3xl bg-primary/12 px-5 py-3 text-[15px] leading-relaxed text-foreground">
                              {m.content}
                            </div>
                          </div>
                        );
                      }
                      if (m.role === 'error') {
                        return (
                          <div key={m.id} className="animate-fade-in flex justify-start">
                            <div className="max-w-[min(100%,40rem)] rounded-2xl border border-destructive/25 bg-destructive/10 px-5 py-3 text-sm text-destructive">
                              {m.content}
                            </div>
                          </div>
                        );
                      }
                      return (
                        <div key={m.id} className="animate-fade-in flex justify-start">
                          <div className="max-w-[min(100%,40rem)] space-y-2">
                            <article className="tech-markdown-doc prd-rag-md text-[15px] leading-relaxed text-foreground">
                              <ReactMarkdown remarkPlugins={[remarkGfm]} components={techMarkdownComponents}>
                                {m.content}
                              </ReactMarkdown>
                            </article>
                            {m.meta?.corrected_question ? (
                              <p className="text-xs text-muted-foreground">
                                <em>Retrieval query:</em> {m.meta.corrected_question}
                              </p>
                            ) : null}
                          </div>
                        </div>
                      );
                    })}
                    <div ref={ragChatEndRef} />
                  </div>
                )}
              </div>
            </div>

            <div className="shrink-0 border-t border-border/50 bg-background/95 px-4 py-3 backdrop-blur-md sm:px-6">
              <div className="mx-auto w-full max-w-4xl space-y-3 lg:max-w-5xl xl:max-w-6xl">
                {ragLastFileName ? (
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="inline-flex max-w-full items-center gap-2 truncate rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
                      <FileText className="h-3.5 w-3.5 shrink-0" fill="currentColor" />
                      <span className="truncate">{ragLastFileName}</span>
                      {ragIndexed ? (
                        <span className="shrink-0 text-[10px] uppercase tracking-wide text-primary/80">Indexed</span>
                      ) : null}
                    </span>
                    {ragBusy ? (
                      <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
                        <Loader2 className="h-3.5 w-3.5 animate-spin" /> Indexing…
                      </span>
                    ) : null}
                  </div>
                ) : null}
                {ragMsg ? (
                  <div className="rounded-xl bg-secondary/50 px-3 py-2 text-xs text-muted-foreground">{ragMsg}</div>
                ) : null}

                <div className="flex items-end gap-2 rounded-[1.75rem] border border-border/60 bg-secondary/25 p-2 shadow-sm focus-within:border-primary/30 focus-within:ring-2 focus-within:ring-primary/15">
                  <button
                    type="button"
                    title="Attach PDF or document"
                    disabled={ragBusy}
                    onClick={() => ragFileInputRef.current?.click()}
                    className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground disabled:pointer-events-none disabled:opacity-40"
                  >
                    <Plus className="h-6 w-6 stroke-[2]" aria-hidden />
                    <span className="sr-only">Attach document</span>
                  </button>
                  <textarea
                    rows={1}
                    value={ragQuestion}
                    onChange={(e) => setRagQuestion(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        if (!ragBusy && ragIndexed) handleRagQuery();
                      }
                    }}
                    placeholder={
                      ragIndexed ? 'Ask anything…' : 'Attach a document with + to start'
                    }
                    disabled={!ragIndexed}
                    className="max-h-40 min-h-[44px] flex-1 resize-none bg-transparent py-3 pr-2 text-[15px] leading-snug text-foreground placeholder:text-muted-foreground focus:outline-none disabled:cursor-not-allowed disabled:opacity-50"
                  />
                  <Button
                    variant="warm"
                    size="icon"
                    className="h-11 w-11 shrink-0 rounded-full"
                    onClick={handleRagQuery}
                    disabled={ragBusy || !ragIndexed || !ragQuestion.trim()}
                  >
                    {ragBusy ? <Loader2 className="h-5 w-5 animate-spin" /> : <Send className="h-5 w-5" />}
                  </Button>
                </div>
                <p className="text-center text-[11px] text-muted-foreground">
                  Enter sends · Shift+Enter newline · BM25 + Chroma + LangGraph (in-memory index on this server)
                </p>
              </div>
            </div>
          </div>
        ) : (
          <>
            <StepIndicator current={step} steps={STEPS} />

            <div className="min-h-[320px]">
              {serviceTab === 'prd' && step === 'capability' && (
                <div className="animate-fade-in space-y-4">
                  <div>
                    <h3 className="font-display text-lg font-semibold text-foreground">Choose a capability</h3>
                    <p className="text-sm text-muted-foreground">Select the type of document you want to generate</p>
                  </div>
                  <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                    {capabilities.map((c) => (
                      <button
                        key={c.id}
                        type="button"
                        onClick={() => selectCapability(c.id)}
                        className="group flex items-start gap-3 rounded-2xl border border-border/60 bg-card/50 p-4 text-left transition-all duration-300 hover:border-primary/30 hover:bg-card hover:shadow-md active:scale-[0.98]"
                      >
                        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary transition-colors group-hover:bg-primary/15">
                          {c.id === 'code_summarizer' ? (
                            <Code2 className="h-5 w-5" fill="currentColor" aria-hidden />
                          ) : (
                            <FileText className="h-5 w-5" fill="currentColor" aria-hidden />
                          )}
                        </div>
                        <div className="min-w-0 flex-1">
                          <p className="font-display text-sm font-semibold text-foreground">{c.title}</p>
                          <p className="mt-0.5 text-xs text-muted-foreground line-clamp-3">{c.description}</p>
                          {c.requires_codebase ? (
                            <span className="mt-1.5 inline-flex items-center gap-1 rounded-full bg-accent/10 px-2 py-0.5 text-[10px] font-medium text-accent">
                              <FolderArchive className="h-2.5 w-2.5" /> ZIP required
                            </span>
                          ) : null}
                        </div>
                        <ChevronRight className="mt-1 h-4 w-4 shrink-0 text-muted-foreground/40 transition-transform group-hover:translate-x-0.5 group-hover:text-primary/60" aria-hidden />
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {serviceTab === 'prd' && step === 'questions' && selectedCap && selectedCap === 'code_summarizer' && (
                <div className="animate-fade-in space-y-5">
                  <div>
                    <h3 className="font-display flex items-center gap-2 text-lg font-semibold text-foreground">
                      <Code2 className="h-5 w-5 text-primary" aria-hidden /> Full project code summary
                    </h3>
                    <p className="mt-2 text-sm text-muted-foreground">
                      Summarizes your <strong className="text-foreground">indexed</strong> repository (purpose, tech stack,
                      architecture, key modules, flows). Upload a ZIP first or paste a{' '}
                      <code className="rounded bg-secondary px-1 py-0.5 text-xs">project_id</code> you already have.
                    </p>
                    <p className="mt-3 rounded-xl border border-primary/20 bg-primary/5 px-4 py-3 text-sm text-muted-foreground">
                      No pasted source needed—the model uses RAG over chunks from your index.
                    </p>
                  </div>
                  <div className="space-y-4">
                    <label className="space-y-1.5">
                      <span className="text-sm font-medium text-foreground">Detail level</span>
                      <select
                        className={inputFieldClass}
                        value={csDetail}
                        onChange={(e) => setCsDetail(e.target.value)}
                      >
                        <option value="short">Short</option>
                        <option value="medium">Medium</option>
                        <option value="detailed">Detailed</option>
                      </select>
                    </label>
                    <label className="space-y-1.5">
                      <span className="text-sm font-medium text-foreground">Indexed project ID</span>
                      <input
                        type="text"
                        className={inputFieldClass}
                        value={projectId}
                        onChange={(e) => {
                          setProjectId(e.target.value);
                          setStoredProjectId(e.target.value);
                        }}
                        placeholder="UUID from Upload / index ZIP"
                      />
                    </label>
                  </div>

                  {genError ? <p className="text-sm text-destructive">{genError}</p> : null}
                  <div className="flex flex-wrap items-center gap-3 pt-2">
                    <Button variant="warm-outline" size="sm" onClick={goCapability}>
                      <ChevronLeft className="h-4 w-4" /> Back
                    </Button>
                    <Button variant="warm-outline" size="sm" onClick={() => setStep('upload')}>
                      <FileArchive className="h-4 w-4" />
                      Upload / index ZIP
                    </Button>
                    <Button variant="warm" size="sm" disabled={generating} onClick={handleCodeSummarizer}>
                      {generating ? (
                        <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
                      ) : (
                        <Zap className="h-4 w-4" fill="currentColor" />
                      )}
                      {generating ? 'Summarizing…' : 'Generate full summary'}
                    </Button>
                  </div>
                </div>
              )}

              {serviceTab === 'prd' && step === 'questions' && selectedCap && selectedCap !== 'code_summarizer' && (
                <div className="animate-fade-in space-y-5">
                  <div>
                    <h3 className="font-display text-lg font-semibold text-foreground">Clarifying questions</h3>
                    <p className="text-sm text-muted-foreground">Fill in the required details below</p>
                  </div>
                  {currentQuestions.length === 0 ? (
                    <p className="text-sm text-muted-foreground">No extra questions for this capability. Continue.</p>
                  ) : (
                    <div className="max-h-[400px] space-y-4 overflow-y-auto pr-1">
                      {currentQuestions.map((q) => (
                        <div key={q.id} className="space-y-1.5">
                          <label className="text-sm font-medium text-foreground">
                            {q.label}
                            {q.required ? <span className="ml-1 text-accent">*</span> : null}
                          </label>
                          {q.type === 'textarea' ? (
                            <textarea
                              className={`${textareaFieldClass} min-h-[96px]`}
                              rows={4}
                              value={answers[q.id] || ''}
                              onChange={(e) => handleAnswerChange(q.id, e.target.value)}
                              placeholder={q.placeholder || ''}
                            />
                          ) : (
                            <input
                              type="text"
                              className={inputFieldClass}
                              value={answers[q.id] || ''}
                              onChange={(e) => handleAnswerChange(q.id, e.target.value)}
                              placeholder={q.placeholder || ''}
                            />
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                  {genError ? <p className="text-sm text-destructive">{genError}</p> : null}
                  <div className="flex flex-wrap items-center gap-3 pt-2">
                    <Button variant="warm-outline" size="sm" onClick={goCapability}>
                      <ChevronLeft className="h-4 w-4" /> Back
                    </Button>
                    <Button variant="warm" size="sm" onClick={handleNextFromQuestions}>
                      Continue <ChevronRight className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              )}

              {serviceTab === 'prd' && step === 'upload' && (
                <div className="animate-fade-in space-y-5">
                  <div>
                    <h3 className="font-display text-lg font-semibold text-foreground">Upload project</h3>
                    <p className="text-sm text-muted-foreground">Upload your project as a .zip file for indexing</p>
                  </div>
                  {selectedCap === 'code_summarizer' ? (
                    <p className="text-sm text-muted-foreground">
                      After indexing, you return to <strong className="text-foreground">Full project code summary</strong> with{' '}
                      <code className="rounded bg-secondary px-1 py-0.5 text-xs">project_id</code> set.
                    </p>
                  ) : null}
                  <p className="text-sm text-muted-foreground">
                    Code is chunked, embedded (local GGUF or Azure depending on server config), then indexed with FAISS. Large
                    folders like <code className="rounded bg-secondary px-1 py-0.5 text-xs">node_modules</code> are skipped.
                    Indexing can take several minutes for large repos.
                  </p>
                  {projectId ? (
                    <p className="text-xs text-muted-foreground">
                      Current indexed project: <code className="rounded bg-secondary px-1.5 py-0.5">{projectId.slice(0, 8)}…</code>{' '}
                      <Button variant="ghost" size="sm" className="h-7 px-2 text-xs" onClick={clearProject}>
                        Clear
                      </Button>
                    </p>
                  ) : null}
                  <div className="space-y-2">
                    <input
                      ref={zipFileInputRef}
                      type="file"
                      accept=".zip,application/zip"
                      className="hidden"
                      onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                    />
                    <div
                      role="button"
                      tabIndex={0}
                      onClick={() => zipFileInputRef.current?.click()}
                      onKeyDown={(e) => e.key === 'Enter' && zipFileInputRef.current?.click()}
                      className="flex cursor-pointer flex-col items-center gap-3 rounded-2xl border-2 border-dashed border-border/60 bg-secondary/20 p-8 text-center transition-all duration-300 hover:border-primary/40 hover:bg-secondary/40"
                    >
                      <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                        <FolderArchive className="h-6 w-6" fill="currentColor" />
                      </div>
                      {uploadFile ? (
                        <div className="space-y-1">
                          <p className="text-sm font-medium text-foreground">{uploadFile.name}</p>
                          <p className="text-xs text-muted-foreground">Click to replace</p>
                        </div>
                      ) : (
                        <div className="space-y-1">
                          <p className="text-sm font-medium text-foreground">Drop your .zip file here</p>
                          <p className="text-xs text-muted-foreground">or click to browse</p>
                        </div>
                      )}
                    </div>
                    {uploadFile ? (
                      <Button variant="ghost" size="sm" className="text-xs text-muted-foreground" onClick={() => setUploadFile(null)}>
                        <Trash2 className="h-3 w-3" /> Clear file
                      </Button>
                    ) : null}
                  </div>
                  {uploadMsg ? <div className="rounded-xl bg-secondary/40 px-4 py-2 text-sm text-muted-foreground">{uploadMsg}</div> : null}
                  <div className="flex flex-wrap items-center gap-3">
                    <Button variant="warm-outline" size="sm" onClick={() => setStep('questions')}>
                      <ChevronLeft className="h-4 w-4" /> Back
                    </Button>
                    <Button variant="warm" size="sm" disabled={uploading} onClick={handleUpload}>
                      {uploading ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : <Upload className="h-4 w-4" />}
                      {uploading ? 'Indexing…' : 'Upload & index'}
                    </Button>
                  </div>
                </div>
              )}

              {serviceTab === 'prd' && step === 'generate' && selectedCap !== 'code_summarizer' && (
                <div className="animate-fade-in space-y-5">
                  <div>
                    <h3 className="font-display text-lg font-semibold text-foreground">Ready to generate</h3>
                    <p className="text-sm text-muted-foreground">Review your inputs and generate</p>
                  </div>
                  <div className="rounded-2xl border border-border/50 bg-secondary/30 p-5">
                    <div className="space-y-2 text-sm">
                      <p>
                        <span className="font-medium text-foreground">Capability:</span>{' '}
                        <span className="text-muted-foreground">{capabilities.find((c) => c.id === selectedCap)?.title}</span>
                      </p>
                      {needsUpload ? (
                        <p>
                          <span className="font-medium text-foreground">Codebase:</span>{' '}
                          {projectId ? (
                            <code className="rounded bg-card px-1.5 py-0.5 text-xs">{projectId.slice(0, 8)}…</code>
                          ) : (
                            <span className="text-destructive">not indexed</span>
                          )}
                        </p>
                      ) : null}
                    </div>
                  </div>
                  {genError ? <p className="text-sm text-destructive">{genError}</p> : null}
                  <div className="flex flex-wrap items-center gap-3">
                    {!needsUpload ? (
                      <Button variant="warm-outline" size="sm" onClick={() => setStep('questions')}>
                        <ChevronLeft className="h-4 w-4" /> Back
                      </Button>
                    ) : (
                      <Button variant="warm-outline" size="sm" onClick={() => setStep('upload')}>
                        <ChevronLeft className="h-4 w-4" /> Back to upload
                      </Button>
                    )}
                    <Button variant="warm" size="sm" disabled={generating} onClick={handleGenerate}>
                      {generating ? (
                        <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
                      ) : (
                        <Zap className="h-4 w-4" fill="currentColor" />
                      )}
                      {generating ? 'Generating…' : 'Generate'}
                    </Button>
                  </div>
                  {generating ? (
                    <p className="text-xs leading-relaxed text-muted-foreground">
                      Local models can take several minutes for long documents. Keep this tab open—the request timeout is 10
                      minutes. Backend logs show <code className="rounded bg-secondary px-1 text-[11px]">PRD platform generate started</code>{' '}
                      then <code className="rounded bg-secondary px-1 text-[11px]">finished</code> when done.
                    </p>
                  ) : null}
                </div>
              )}

              {serviceTab === 'prd' && step === 'result' && result && (
                <div className="animate-fade-in space-y-5">
                  <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="h-5 w-5 text-primary" fill="currentColor" aria-hidden />
                      <h3 className="font-display text-lg font-semibold text-foreground">Document generated</h3>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <Button variant="warm" size="sm" onClick={downloadText} disabled={!result.markdown}>
                        <Download className="h-4 w-4" /> {isPrdPlain ? 'Download text' : 'Download Markdown'}
                      </Button>
                      {canDownloadPdf ? (
                        <Button variant="warm-outline" size="sm" onClick={downloadPdf}>
                          <FileText className="h-4 w-4" /> Download PDF
                        </Button>
                      ) : null}
                      {hasDiagramDownload ? (
                        <Button variant="warm-outline" size="sm" onClick={downloadDiagram}>
                          <ImageDown className="h-4 w-4" /> {result.svg_base64 ? 'Download SVG' : 'Download PNG'}
                        </Button>
                      ) : null}
                      <Button variant="glass" size="sm" onClick={resetFlow}>
                        <RotateCcw className="h-4 w-4" /> New run
                      </Button>
                    </div>
                  </div>
                  {result.message ? <p className="prd-result-message">{result.message}</p> : null}
                  {result.svg_base64 ? (
                    <div className="prd-diagram-frame">
                      <img src={`data:image/svg+xml;base64,${result.svg_base64}`} alt="Diagram" />
                    </div>
                  ) : null}
                  {result.png_base64 ? (
                    <div className="prd-png-wrap">
                      <img src={`data:image/png;base64,${result.png_base64}`} alt="Diagram" />
                    </div>
                  ) : null}
                  <div className="rounded-2xl border border-border/50 bg-card p-5">
                    {prdParsed ? (
                      <article className="prd-document !mt-0 !max-h-none border-0 bg-transparent p-0 shadow-none">
                        <h1 className="prd-document-title">{prdParsed.title}</h1>
                        {prdParsed.sections.map((sec) => (
                          <section key={sec.heading} className="prd-document-section">
                            <h2 className="prd-document-section-title font-display">{formatSectionHeading(sec.heading)}</h2>
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
                      <article className="tech-markdown-doc prd-result-article !mt-0 !max-h-none">
                        <ReactMarkdown remarkPlugins={[remarkGfm]} components={techMarkdownComponents}>
                          {result.markdown || '_No summary returned._'}
                        </ReactMarkdown>
                        {result.csMetadata && Object.keys(result.csMetadata).length > 0 ? (
                          <>
                            <h3 className="prd-document-section-title prd-metadata-heading font-display">Metadata</h3>
                            <pre className="prd-metadata-pre">{JSON.stringify(result.csMetadata, null, 2)}</pre>
                          </>
                        ) : null}
                      </article>
                    ) : isMarkdownDocCap ? (
                      <article className="tech-markdown-doc prd-result-article !mt-0 !max-h-none">
                        <ReactMarkdown remarkPlugins={[remarkGfm]} components={techMarkdownComponents}>
                          {result.markdown || ''}
                        </ReactMarkdown>
                      </article>
                    ) : (
                      <pre className="prd-output-pre !mt-0 rounded-xl bg-secondary/50">{result.markdown || '(no text)'}</pre>
                    )}
                  </div>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
