import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import axios from 'axios';
import { jsPDF } from 'jspdf';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  FileArchive,
  Loader2,
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
  Paperclip,
  FolderArchive,
  Zap,
  CheckCircle2,
  RotateCcw,
  X,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import ServicePlusMenu from '@/components/ServicePlusMenu';
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

const STORAGE_CHATBOT_THREAD_ID = 'docu-atlas-chatbot-thread-id';
const STORAGE_RAG_DOC_ID = 'docu-atlas-rag-document-id';
/** Roll up conversation context after this many user+assistant messages since the last summary. */
const ROLLUP_MESSAGE_THRESHOLD = 15;

function getOrCreateRagDocumentId() {
  try {
    const s = localStorage.getItem(STORAGE_RAG_DOC_ID);
    if (s?.trim()) return s.trim();
    const id = makeRagSessionId();
    localStorage.setItem(STORAGE_RAG_DOC_ID, id);
    return id;
  } catch {
    return makeRagSessionId();
  }
}

function getStoredOrCreateChatbotThreadId() {
  try {
    const s = localStorage.getItem(STORAGE_CHATBOT_THREAD_ID);
    if (s?.trim()) return s.trim();
    const id =
      typeof crypto !== 'undefined' && crypto.randomUUID
        ? crypto.randomUUID()
        : `cb_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`;
    localStorage.setItem(STORAGE_CHATBOT_THREAD_ID, id);
    return id;
  } catch {
    return `cb_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`;
  }
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

const inputFieldClass =
  'w-full rounded-xl border border-input bg-card px-4 py-2.5 text-sm text-foreground shadow-sm transition-colors placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20';

const textareaFieldClass = `${inputFieldClass} resize-none`;

function deriveDocPresentation(capabilityId, result) {
  if (!result) return null;
  const isPrdPlain =
    Boolean(result.markdown) && (capabilityId === 'prd' || result.content_format === 'prd_plain');
  const prdParsed = isPrdPlain ? parsePrdPlain(result.markdown) : null;
  const isCodeSummarizer =
    capabilityId === 'code_summarizer' || result.content_format === 'code_summarizer';
  const isMarkdownDocCap = Boolean(
    capabilityId && MARKDOWN_DOC_CAPS.has(capabilityId) && result.markdown,
  );
  const markdownPdfParsed =
    isMarkdownDocCap && result.markdown ? parseMarkdownForPdf(result.markdown) : null;
  const canDownloadPdf = Boolean(prdParsed || markdownPdfParsed);
  const hasDiagramDownload = Boolean(result.svg_base64 || result.png_base64);
  return {
    isPrdPlain,
    prdParsed,
    isCodeSummarizer,
    isMarkdownDocCap,
    markdownPdfParsed,
    canDownloadPdf,
    hasDiagramDownload,
  };
}

function downloadTextForResult(result, capabilityId) {
  if (!result?.markdown) return;
  const isPrd = capabilityId === 'prd' || result.content_format === 'prd_plain';
  const isCs = result.content_format === 'code_summarizer';
  const blob = new Blob([result.markdown], {
    type: isPrd ? 'text/plain;charset=utf-8' : 'text/markdown;charset=utf-8',
  });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  let name = `${capabilityId || 'output'}.md`;
  if (isPrd) name = `${capabilityId || 'prd'}.txt`;
  else if (isCs) name = 'full_project_code_summary.md';
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

function downloadPdfForResult(result, capabilityId) {
  if (!result?.markdown) return;
  const isPrdPlain = capabilityId === 'prd' || result.content_format === 'prd_plain';
  const prdParsed = isPrdPlain ? parsePrdPlain(result.markdown) : null;
  const isMarkdownDocCap =
    capabilityId && MARKDOWN_DOC_CAPS.has(capabilityId) && result.markdown;
  const markdownParsed =
    isMarkdownDocCap && result.markdown ? parseMarkdownForPdf(result.markdown) : null;
  if (!prdParsed && !markdownParsed) return;

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
    const { title, parts } = markdownParsed;
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

  doc.save(`${capabilityId || 'document'}.pdf`);
}

function downloadDiagramForResult(result, capabilityId) {
  if (!result) return;
  const base = capabilityId || 'diagram';
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
}

function DocResultBody({ capabilityId, result }) {
  const pres = deriveDocPresentation(capabilityId, result);
  if (!pres || !result) return null;
  const {
    prdParsed,
    isCodeSummarizer,
    isMarkdownDocCap,
  } = pres;

  return (
    <div className="w-full rounded-2xl border border-border/50 bg-card p-6 sm:p-8 lg:p-10">
      {result.svg_base64 ? (
        <div className="prd-diagram-frame mb-4">
          <img src={`data:image/svg+xml;base64,${result.svg_base64}`} alt="Diagram" />
        </div>
      ) : null}
      {result.png_base64 ? (
        <div className="prd-png-wrap mb-4">
          <img src={`data:image/png;base64,${result.png_base64}`} alt="Diagram" />
        </div>
      ) : null}
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
  );
}

export default function PRDPlatformPanel() {
  const [panelMode, setPanelMode] = useState('hybrid_rag');
  const [plusMenuOpen, setPlusMenuOpen] = useState(false);
  const plusButtonRef = useRef(null);
  const [docMessages, setDocMessages] = useState([]);
  const docChatEndRef = useRef(null);

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

  const [csDetail, setCsDetail] = useState('medium');

  const [ragDocumentId, setRagDocumentId] = useState(() => getOrCreateRagDocumentId());
  const [ragIndexed, setRagIndexed] = useState(false);
  const [ragLastFileName, setRagLastFileName] = useState('');
  const [ragQuestion, setRagQuestion] = useState('');
  const [ragBusy, setRagBusy] = useState(false);
  const [ragMsg, setRagMsg] = useState('');
  const [ragChat, setRagChat] = useState([]);
  const [ragConversationSummary, setRagConversationSummary] = useState('');
  const [ragRollupMessageIndex, setRagRollupMessageIndex] = useState(0);
  const [ragHydrated, setRagHydrated] = useState(false);
  const [ragDropActive, setRagDropActive] = useState(false);
  const ragChatEndRef = useRef(null);
  const ragFileInputRef = useRef(null);
  const zipFileInputRef = useRef(null);
  const ragRollupMessageIndexRef = useRef(0);
  const ragSummaryRef = useRef('');

  const [chatbotThreadId, setChatbotThreadId] = useState(() => getStoredOrCreateChatbotThreadId());
  const [chatbotMessages, setChatbotMessages] = useState([]);
  const [chatbotInput, setChatbotInput] = useState('');
  const [chatbotBusy, setChatbotBusy] = useState(false);
  const [chatbotConversationSummary, setChatbotConversationSummary] = useState('');
  const [chatbotRollupMessageIndex, setChatbotRollupMessageIndex] = useState(0);
  const [chatbotHydrated, setChatbotHydrated] = useState(false);
  const chatbotChatEndRef = useRef(null);
  const chatbotRollupMessageIndexRef = useRef(0);
  const chatbotSummaryRef = useRef('');

  useEffect(() => {
    ragChatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [ragChat]);

  useEffect(() => {
    chatbotChatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatbotMessages]);

  useEffect(() => {
    docChatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [docMessages, selectedCap, panelMode]);

  useEffect(() => {
    ragRollupMessageIndexRef.current = ragRollupMessageIndex;
  }, [ragRollupMessageIndex]);
  useEffect(() => {
    chatbotRollupMessageIndexRef.current = chatbotRollupMessageIndex;
  }, [chatbotRollupMessageIndex]);
  useEffect(() => {
    ragSummaryRef.current = ragConversationSummary;
  }, [ragConversationSummary]);
  useEffect(() => {
    chatbotSummaryRef.current = chatbotConversationSummary;
  }, [chatbotConversationSummary]);

  useEffect(() => {
    let cancelled = false;
    setChatbotHydrated(false);
    (async () => {
      try {
        const { data } = await axios.get(
          `${API_PRD}/chat/threads/${encodeURIComponent(chatbotThreadId)}`,
          { timeout: 60000 },
        );
        if (cancelled) return;
        setChatbotMessages(Array.isArray(data.messages) ? data.messages : []);
        setChatbotConversationSummary(data.conversation_summary || '');
        setChatbotRollupMessageIndex(
          typeof data.rollup_message_index === 'number' ? data.rollup_message_index : 0,
        );
      } catch (e) {
        if (cancelled) return;
        setChatbotMessages([]);
        setChatbotConversationSummary('');
        setChatbotRollupMessageIndex(0);
      } finally {
        if (!cancelled) setChatbotHydrated(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [chatbotThreadId]);

  useEffect(() => {
    let cancelled = false;
    setRagHydrated(false);
    (async () => {
      try {
        const { data } = await axios.get(
          `${API_PRD}/chat/threads/${encodeURIComponent(ragDocumentId)}`,
          { timeout: 60000 },
        );
        if (cancelled) return;
        setRagChat(Array.isArray(data.messages) ? data.messages : []);
        setRagConversationSummary(data.conversation_summary || '');
        setRagRollupMessageIndex(
          typeof data.rollup_message_index === 'number' ? data.rollup_message_index : 0,
        );
        setRagIndexed(Boolean(data.rag_indexed));
        setRagLastFileName(typeof data.rag_last_file_name === 'string' ? data.rag_last_file_name : '');
      } catch (e) {
        if (cancelled) return;
        setRagChat([]);
        setRagConversationSummary('');
        setRagRollupMessageIndex(0);
        if (e.response?.status === 404) {
          setRagIndexed(false);
          setRagLastFileName('');
        }
      } finally {
        if (!cancelled) setRagHydrated(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [ragDocumentId]);

  useEffect(() => {
    if (!ragHydrated) return undefined;
    const t = setTimeout(() => {
      axios
        .put(
          `${API_PRD}/chat/threads/${encodeURIComponent(ragDocumentId)}`,
          {
            mode: 'hybrid_rag',
            rag_document_id: ragDocumentId,
            conversation_summary: ragConversationSummary,
            rollup_message_index: ragRollupMessageIndex,
            rag_indexed: ragIndexed,
            rag_last_file_name: ragLastFileName,
            messages: ragChat.map((m) => ({
              id: m.id,
              role: m.role,
              content: m.content,
              ...(m.meta ? { meta: m.meta } : {}),
            })),
          },
          { timeout: 120000 },
        )
        .catch(() => {});
    }, 300);
    return () => clearTimeout(t);
  }, [
    ragHydrated,
    ragDocumentId,
    ragChat,
    ragConversationSummary,
    ragRollupMessageIndex,
    ragIndexed,
    ragLastFileName,
  ]);

  useEffect(() => {
    if (!chatbotHydrated) return undefined;
    const t = setTimeout(() => {
      axios
        .put(
          `${API_PRD}/chat/threads/${encodeURIComponent(chatbotThreadId)}`,
          {
            mode: 'chatbot',
            conversation_summary: chatbotConversationSummary,
            rollup_message_index: chatbotRollupMessageIndex,
            rag_indexed: false,
            rag_last_file_name: '',
            messages: chatbotMessages.map((m) => ({
              id: m.id,
              role: m.role,
              content: m.content,
            })),
          },
          { timeout: 120000 },
        )
        .catch(() => {});
    }, 300);
    return () => clearTimeout(t);
  }, [chatbotHydrated, chatbotThreadId, chatbotMessages, chatbotConversationSummary, chatbotRollupMessageIndex]);

  const tryRollup = useCallback(async (mode, fullMessages) => {
    const startIdx =
      mode === 'rag' ? ragRollupMessageIndexRef.current : chatbotRollupMessageIndexRef.current;
    const prior = mode === 'rag' ? ragSummaryRef.current : chatbotSummaryRef.current;
    const slice = fullMessages
      .slice(startIdx)
      .filter((m) => m.role === 'user' || m.role === 'assistant');
    if (slice.length < ROLLUP_MESSAGE_THRESHOLD) return;
    const capped = slice.length > 24 ? slice.slice(-24) : slice;
    const exchanges = capped.map((m) => ({ role: m.role, content: m.content }));
    try {
      const { data } = await axios.post(
        `${API_PRD}/conversation/rollup`,
        { prior_summary: prior, exchanges },
        { timeout: 120000 },
      );
      const s = (data.summary || '').trim();
      if (!s) return;
      const endLen = fullMessages.length;
      if (mode === 'rag') {
        setRagConversationSummary(s);
        ragSummaryRef.current = s;
        setRagRollupMessageIndex(endLen);
        ragRollupMessageIndexRef.current = endLen;
      } else {
        setChatbotConversationSummary(s);
        chatbotSummaryRef.current = s;
        setChatbotRollupMessageIndex(endLen);
        chatbotRollupMessageIndexRef.current = endLen;
      }
    } catch {
      /* optional rollup failure */
    }
  }, []);

  const resetCodeSummarizerForm = useCallback(() => {
    setCsDetail('medium');
  }, []);

  useEffect(() => {
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

  const selectedCapabilityTitle = useMemo(() => {
    if (!selectedCap) return '';
    return capabilities.find((x) => x.id === selectedCap)?.title || selectedCap;
  }, [selectedCap, capabilities]);

  const handleAnswerChange = useCallback((key, value) => {
    setAnswers((prev) => ({ ...prev, [key]: value }));
  }, []);

  const validateQuestions = useCallback(() => {
    for (const q of currentQuestions) {
      if (q.required && !(answers[q.id] || '').trim()) return q.label;
    }
    return null;
  }, [currentQuestions, answers]);

  const selectCapabilityFromMenu = useCallback(
    (id) => {
      setPanelMode('document');
      setPlusMenuOpen(false);
      setSelectedCap(id);
      setAnswers({});
      resetCodeSummarizerForm();
      setGenError('');
      setUploadMsg('');
    },
    [resetCodeSummarizerForm],
  );

  const selectHybridRagFromMenu = useCallback(() => {
    setPanelMode('hybrid_rag');
    setPlusMenuOpen(false);
    setSelectedCap(null);
    setAnswers({});
    setGenError('');
  }, []);

  const selectChatbotFromMenu = useCallback(() => {
    setPanelMode('chatbot');
    setPlusMenuOpen(false);
    setSelectedCap(null);
    setAnswers({});
    setGenError('');
  }, []);

  const cancelDocForm = useCallback(() => {
    setSelectedCap(null);
    setAnswers({});
    setGenError('');
    setUploadMsg('');
    setUploadFile(null);
    resetCodeSummarizerForm();
  }, [resetCodeSummarizerForm]);

  const clearDocThread = useCallback(() => {
    setDocMessages([]);
    cancelDocForm();
  }, [cancelDocForm]);

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
    } catch (e) {
      setUploadMsg(e.response?.data?.detail || e.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  }, [uploadFile, projectId]);

  const handleCodeSummarizer = useCallback(async () => {
    setGenError('');
    setGenerating(true);
    const detail_level = csDetail;
    const pid = projectId.trim();
    const capId = selectedCap;
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
      const payload = {
        markdown: data.summary || '',
        message: `Detail level: ${data.level || detail_level}.`,
        content_format: 'code_summarizer',
        csMetadata: data.metadata,
      };
      setDocMessages((m) => [
        ...m,
        { id: Date.now(), role: 'assistant', capabilityId: capId, result: payload },
      ]);
      setSelectedCap(null);
      setAnswers({});
      resetCodeSummarizerForm();
    } catch (e) {
      const detail = e.response?.data?.detail;
      setGenError(
        typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : e.message || 'Summarization failed',
      );
    } finally {
      setGenerating(false);
    }
  }, [csDetail, projectId, selectedCap, resetCodeSummarizerForm]);

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
    const capId = selectedCap;
    try {
      const { data } = await axios.post(
        `${API_PRD}/projects/generate`,
        {
          project_id: projectId || null,
          capability: capId,
          answers,
        },
        { timeout: 600000 },
      );
      setDocMessages((m) => [...m, { id: Date.now(), role: 'assistant', capabilityId: capId, result: data }]);
      setSelectedCap(null);
      setAnswers({});
    } catch (e) {
      setGenError(e.response?.data?.detail || e.message || 'Generation failed');
    } finally {
      setGenerating(false);
    }
  }, [needsUpload, projectId, selectedCap, answers, validateQuestions]);

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
        setRagConversationSummary('');
        setRagRollupMessageIndex(0);
        ragRollupMessageIndexRef.current = 0;
        ragSummaryRef.current = '';
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
    const summaryForRequest = ragConversationSummary.trim();
    try {
      const { data } = await axios.post(
        `${API_PRD_RAG}/query`,
        {
          document_id: ragDocumentId,
          question: q,
          ...(summaryForRequest ? { conversation_summary: summaryForRequest } : {}),
        },
        { timeout: 600000 },
      );
      setRagChat((prev) => {
        const next = [
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
        ];
        void tryRollup('rag', next);
        return next;
      });
      setRagMsg(`Last reply in ${data.elapsed_seconds ?? '?'}s (retries=${data.retries ?? 0}).`);
    } catch (e) {
      const d = e.response?.data?.detail;
      const msg = typeof d === 'string' ? d : d ? JSON.stringify(d) : e.message || 'Query failed';
      setRagChat((prev) => [...prev, { id: userId + 1, role: 'error', content: msg }]);
      setRagMsg('');
    } finally {
      setRagBusy(false);
    }
  }, [ragDocumentId, ragIndexed, ragQuestion, ragConversationSummary, tryRollup]);

  const clearChatbotThread = useCallback(async () => {
    const tid = chatbotThreadId;
    try {
      await axios.delete(`${API_PRD}/chat/threads/${encodeURIComponent(tid)}`);
    } catch {
      /* ignore */
    }
    const newId =
      typeof crypto !== 'undefined' && crypto.randomUUID
        ? crypto.randomUUID()
        : `cb_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`;
    try {
      localStorage.setItem(STORAGE_CHATBOT_THREAD_ID, newId);
    } catch {
      /* ignore */
    }
    setChatbotThreadId(newId);
    setChatbotMessages([]);
    setChatbotInput('');
    setChatbotConversationSummary('');
    setChatbotRollupMessageIndex(0);
    chatbotRollupMessageIndexRef.current = 0;
    chatbotSummaryRef.current = '';
  }, [chatbotThreadId]);

  const handleChatbotSend = useCallback(async () => {
    const text = chatbotInput.trim();
    if (!text || chatbotBusy) return;
    setChatbotBusy(true);
    const history = chatbotMessages
      .filter((m) => m.role === 'user' || m.role === 'assistant')
      .map((m) => ({ role: m.role, content: m.content }));
    const summaryForRequest = chatbotConversationSummary.trim();
    const userId = Date.now();
    setChatbotMessages((prev) => [...prev, { id: userId, role: 'user', content: text }]);
    setChatbotInput('');
    try {
      const { data } = await axios.post(
        `${API_PRD}/chatbot`,
        {
          message: text,
          history,
          ...(summaryForRequest ? { conversation_summary: summaryForRequest } : {}),
        },
        { timeout: 600000 },
      );
      setChatbotMessages((prev) => {
        const next = [
          ...prev,
          { id: userId + 1, role: 'assistant', content: data.answer || '_No answer text returned._' },
        ];
        void tryRollup('chatbot', next);
        return next;
      });
    } catch (e) {
      const d = e.response?.data?.detail;
      const msg = typeof d === 'string' ? d : d ? JSON.stringify(d) : e.message || 'Chat failed';
      setChatbotMessages((prev) => [...prev, { id: userId + 1, role: 'error', content: msg }]);
    } finally {
      setChatbotBusy(false);
    }
  }, [chatbotInput, chatbotBusy, chatbotMessages, chatbotConversationSummary, tryRollup]);

  if (capLoading) {
    return (
      <div className="fixed inset-x-0 bottom-0 top-16 z-40 flex flex-col items-center justify-center bg-background">
        <Loader2 className="h-8 w-8 animate-spin text-primary" aria-hidden />
        <p className="mt-4 text-sm text-muted-foreground">Loading OgesAssistant…</p>
      </div>
    );
  }

  const renderDocAssistantMessage = (m) => {
    const { capabilityId, result: res } = m;
    const pres = deriveDocPresentation(capabilityId, res);
    if (!pres || !res) return null;
    const isPrdPlain =
      capabilityId === 'prd' || res.content_format === 'prd_plain';
    return (
      <div key={m.id} className="animate-fade-in flex w-full min-w-0 justify-start">
        <div className="w-full min-w-0 space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <CheckCircle2 className="h-4 w-4 shrink-0 text-primary" fill="currentColor" aria-hidden />
            <span className="font-display text-sm font-semibold text-foreground">Generated</span>
            <div className="flex flex-wrap gap-2">
              <Button
                variant="warm"
                size="sm"
                onClick={() => downloadTextForResult(res, capabilityId)}
                disabled={!res.markdown}
              >
                <Download className="h-4 w-4" /> {isPrdPlain ? 'Text' : 'Markdown'}
              </Button>
              {pres.canDownloadPdf ? (
                <Button variant="warm-outline" size="sm" onClick={() => downloadPdfForResult(res, capabilityId)}>
                  <FileText className="h-4 w-4" /> PDF
                </Button>
              ) : null}
              {pres.hasDiagramDownload ? (
                <Button variant="warm-outline" size="sm" onClick={() => downloadDiagramForResult(res, capabilityId)}>
                  <ImageDown className="h-4 w-4" /> {res.svg_base64 ? 'SVG' : 'PNG'}
                </Button>
              ) : null}
            </div>
          </div>
          {res.message ? <p className="prd-result-message text-sm">{res.message}</p> : null}
          <DocResultBody capabilityId={capabilityId} result={res} />
        </div>
      </div>
    );
  };

  const zipUploadBlock = (
    <div className="space-y-2 border-t border-border/40 pt-4">
      <p className="text-xs text-muted-foreground">
        Code is chunked and indexed with FAISS. Large folders like{' '}
        <code className="rounded bg-secondary px-1 text-[11px]">node_modules</code> are skipped.
      </p>
      {projectId ? (
        <p className="text-xs text-muted-foreground">
          Indexed project: <code className="rounded bg-secondary px-1.5 py-0.5">{projectId.slice(0, 8)}…</code>{' '}
          <Button variant="ghost" size="sm" className="h-7 px-2 text-xs" type="button" onClick={clearProject}>
            Clear
          </Button>
        </p>
      ) : null}
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
        className="flex cursor-pointer flex-col items-center gap-3 rounded-xl border-2 border-dashed border-border/60 bg-secondary/20 py-10 text-center transition-colors hover:border-primary/40 sm:py-14"
      >
        <FolderArchive className="h-8 w-8 text-primary" fill="currentColor" aria-hidden />
        {uploadFile ? (
          <p className="text-sm font-medium text-foreground">{uploadFile.name}</p>
        ) : (
          <p className="text-sm text-muted-foreground">Drop .zip or click to browse</p>
        )}
      </div>
      {uploadFile ? (
        <Button variant="ghost" size="sm" type="button" className="text-xs" onClick={() => setUploadFile(null)}>
          <Trash2 className="h-3 w-3" /> Clear file
        </Button>
      ) : null}
      {uploadMsg ? <div className="rounded-lg bg-secondary/40 px-3 py-2 text-xs text-muted-foreground">{uploadMsg}</div> : null}
      <Button variant="warm-outline" size="sm" type="button" disabled={uploading} onClick={handleUpload}>
        {uploading ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : <Upload className="h-4 w-4" />}
        {uploading ? 'Indexing…' : 'Upload & index'}
      </Button>
    </div>
  );

  const documentFormCard =
    panelMode === 'document' && selectedCap ? (
      <div className="animate-fade-in w-full min-w-0 pb-6">
        <div className="rounded-2xl border border-primary/25 bg-card/90 p-6 shadow-md sm:p-8 lg:p-10">
          <div className="mb-6 flex items-start justify-between gap-4">
            <div>
              <h3 className="font-display text-lg font-semibold text-foreground sm:text-xl">{selectedCapabilityTitle}</h3>
              <p className="mt-2 max-w-4xl text-sm text-muted-foreground">
                {capabilities.find((c) => c.id === selectedCap)?.description}
              </p>
            </div>
            <button
              type="button"
              onClick={cancelDocForm}
              className="rounded-lg p-1.5 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
              aria-label="Close form"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {selectedCap === 'code_summarizer' ? (
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Uses RAG over indexed chunks. Upload a ZIP below or paste a{' '}
                <code className="rounded bg-secondary px-1 text-xs">project_id</code>.
              </p>
              <label className="block space-y-1.5">
                <span className="text-sm font-medium">Detail level</span>
                <select className={inputFieldClass} value={csDetail} onChange={(e) => setCsDetail(e.target.value)}>
                  <option value="short">Short</option>
                  <option value="medium">Medium</option>
                  <option value="detailed">Detailed</option>
                </select>
              </label>
              <label className="block space-y-1.5">
                <span className="text-sm font-medium">Indexed project ID</span>
                <input
                  type="text"
                  className={inputFieldClass}
                  value={projectId}
                  onChange={(e) => {
                    setProjectId(e.target.value);
                    setStoredProjectId(e.target.value);
                  }}
                  placeholder="UUID from index"
                />
              </label>
              {zipUploadBlock}
              {genError ? <p className="text-sm text-destructive">{genError}</p> : null}
              <Button variant="warm" size="sm" type="button" disabled={generating} onClick={handleCodeSummarizer}>
                {generating ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : <Zap className="h-4 w-4" fill="currentColor" />}
                {generating ? 'Summarizing…' : 'Generate full summary'}
              </Button>
            </div>
          ) : (
            <div className="space-y-4">
              {currentQuestions.length === 0 ? (
                <p className="text-sm text-muted-foreground">No extra questions. Generate when ready.</p>
              ) : (
                <div className="max-h-[min(72vh,48rem)] space-y-4 overflow-y-auto pr-1">
                  {currentQuestions.map((q) => (
                    <div key={q.id} className="space-y-2">
                      <label className="text-sm font-medium text-foreground sm:text-base">
                        {q.label}
                        {q.required ? <span className="ml-1 text-accent">*</span> : null}
                      </label>
                      {q.type === 'textarea' ? (
                        <textarea
                          className={`${textareaFieldClass} min-h-[120px] text-[15px] sm:min-h-[140px]`}
                          rows={5}
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
              {needsUpload ? zipUploadBlock : null}
              {genError ? <p className="text-sm text-destructive">{genError}</p> : null}
              <Button variant="warm" size="sm" type="button" disabled={generating} onClick={handleGenerate}>
                {generating ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : <Zap className="h-4 w-4" fill="currentColor" />}
                {generating ? 'Generating…' : 'Generate'}
              </Button>
              {generating ? (
                <p className="text-xs text-muted-foreground">
                  Local models may take several minutes. Request timeout is 10 minutes.
                </p>
              ) : null}
            </div>
          )}
        </div>
      </div>
    ) : null;

  return (
    <div className="fixed inset-x-0 bottom-0 top-16 z-40 flex flex-col bg-background">
      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-border/40 bg-card/80 px-4 py-2.5 backdrop-blur-xl">
        <span className="text-sm font-medium text-muted-foreground">
          {panelMode === 'hybrid_rag'
            ? 'Hybrid RAG'
            : panelMode === 'chatbot'
              ? 'Chatbot'
              : 'Document generation'}
        </span>
        <div className="flex-1" />
        {panelMode === 'document' && docMessages.length > 0 ? (
          <Button variant="ghost" size="sm" className="text-muted-foreground" type="button" onClick={clearDocThread}>
            <RotateCcw className="h-4 w-4" /> New conversation
          </Button>
        ) : null}
        {panelMode === 'hybrid_rag' ? (
          <Button
            variant="ghost"
            size="sm"
            type="button"
            className="text-muted-foreground"
            onClick={async () => {
              const oldId = ragDocumentId;
              const newId = makeRagSessionId();
              try {
                await axios.delete(`${API_PRD}/chat/threads/${encodeURIComponent(oldId)}`);
              } catch {
                /* ignore */
              }
              try {
                localStorage.setItem(STORAGE_RAG_DOC_ID, newId);
              } catch {
                /* ignore */
              }
              setRagDocumentId(newId);
              setRagChat([]);
              setRagMsg('');
              setRagLastFileName('');
              setRagIndexed(false);
              setRagConversationSummary('');
              setRagRollupMessageIndex(0);
              ragRollupMessageIndexRef.current = 0;
              ragSummaryRef.current = '';
            }}
          >
            <Trash2 className="h-4 w-4" /> New chat
          </Button>
        ) : null}
        {panelMode === 'chatbot' ? (
          <>
            <Button variant="ghost" size="sm" type="button" className="text-muted-foreground" onClick={clearChatbotThread}>
              <Trash2 className="h-4 w-4" /> New chat
            </Button>
            <Button variant="outline" size="sm" type="button" onClick={() => setPanelMode('hybrid_rag')}>
              <Search className="h-3.5 w-3.5" fill="currentColor" /> Hybrid RAG
            </Button>
          </>
        ) : null}
        {panelMode === 'document' ? (
          <Button variant="outline" size="sm" type="button" onClick={() => setPanelMode('hybrid_rag')}>
            <Search className="h-3.5 w-3.5" fill="currentColor" /> Hybrid RAG
          </Button>
        ) : null}
      </div>

      {capError ? (
        <div className="shrink-0 border-b border-destructive/20 bg-destructive/10 px-4 py-2 text-center text-xs text-destructive">
          {capError} — document types below may be unavailable until the API recovers.
        </div>
      ) : null}

      {panelMode === 'hybrid_rag' ? (
        <div className="flex min-h-0 flex-1 flex-col">
          <input
            ref={ragFileInputRef}
            type="file"
            accept={RAG_FILE_ACCEPT}
            className="hidden"
            onChange={onRagFileInputChange}
          />
          <div
            className={`relative flex min-h-0 flex-1 flex-col overflow-y-auto ${ragDropActive ? 'bg-primary/5' : ''}`}
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
              <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center border-2 border-dashed border-primary/50 bg-primary/10">
                <p className="text-sm font-medium text-primary">Drop file to index</p>
              </div>
            ) : null}
            <div className="mx-auto flex min-h-full w-full max-w-4xl flex-1 flex-col px-3 py-4 lg:max-w-5xl xl:max-w-6xl xl:px-8">
              {ragChat.length === 0 ? (
                <div className="flex flex-1 flex-col items-center justify-center pb-32 text-center">
                  <MessageSquare className="mb-3 h-10 w-10 text-muted-foreground/35" strokeWidth={1.25} aria-hidden />
                  <p className="max-w-sm text-sm text-muted-foreground">
                    {ragIndexed
                      ? 'Ask a question below.'
                      : 'Attach a document with the paperclip, then ask a question. Use + for document types or Hybrid RAG.'}
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
                          <Button
                            variant="outline"
                            size="sm"
                            type="button"
                            className="text-xs"
                            onClick={() => {
                              const blob = new Blob([m.content], { type: 'text/markdown;charset=utf-8' });
                              const a = document.createElement('a');
                              a.href = URL.createObjectURL(blob);
                              a.download = 'rag-reply.md';
                              a.click();
                              URL.revokeObjectURL(a.href);
                            }}
                          >
                            <Download className="h-3.5 w-3.5" /> Download reply
                          </Button>
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
                      <Loader2 className="h-3.5 w-3.5 animate-spin" /> Working…
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
                  title="Attach document"
                  disabled={ragBusy}
                  onClick={() => ragFileInputRef.current?.click()}
                  className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground disabled:pointer-events-none disabled:opacity-40"
                >
                  <Paperclip className="h-5 w-5" aria-hidden />
                  <span className="sr-only">Attach document</span>
                </button>
                <div className="relative shrink-0">
                  <ServicePlusMenu
                    isOpen={plusMenuOpen}
                    onClose={() => setPlusMenuOpen(false)}
                    onSelectChatbot={selectChatbotFromMenu}
                    onSelectCapability={selectCapabilityFromMenu}
                    onSelectHybridRag={selectHybridRagFromMenu}
                    capabilities={capabilities}
                    capabilitiesDisabled={Boolean(capError)}
                    anchorRef={plusButtonRef}
                  />
                  <button
                    ref={plusButtonRef}
                    type="button"
                    title="Services menu"
                    onClick={() => setPlusMenuOpen((v) => !v)}
                    className="flex h-11 w-11 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
                  >
                    <Plus className="h-6 w-6 stroke-[2]" aria-hidden />
                    <span className="sr-only">Open services</span>
                  </button>
                </div>
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
                  placeholder={ragIndexed ? 'Ask anything…' : 'Attach a file with the paperclip first'}
                  disabled={!ragIndexed}
                  className="max-h-40 min-h-[44px] flex-1 resize-none bg-transparent py-3 pr-2 text-[15px] leading-snug text-foreground placeholder:text-muted-foreground focus:outline-none disabled:cursor-not-allowed disabled:opacity-50"
                />
                <Button
                  variant="warm"
                  size="icon"
                  className="h-11 w-11 shrink-0 rounded-full"
                  type="button"
                  onClick={handleRagQuery}
                  disabled={ragBusy || !ragIndexed || !ragQuestion.trim()}
                >
                  {ragBusy ? <Loader2 className="h-5 w-5 animate-spin" /> : <Send className="h-5 w-5" />}
                </Button>
              </div>
              <p className="text-center text-[11px] text-muted-foreground">
                Enter sends · Shift+Enter newline · BM25 + Chroma + LangGraph
              </p>
            </div>
          </div>
        </div>
      ) : panelMode === 'chatbot' ? (
        <div className="flex min-h-0 flex-1 flex-col">
          <div className="relative flex min-h-0 flex-1 flex-col overflow-y-auto">
            <div className="mx-auto flex min-h-full w-full max-w-4xl flex-1 flex-col px-3 py-4 lg:max-w-5xl xl:max-w-6xl xl:px-8">
              {chatbotMessages.length === 0 ? (
                <div className="flex flex-1 flex-col items-center justify-center pb-32 text-center">
                  <MessageSquare className="mb-3 h-10 w-10 text-muted-foreground/35" strokeWidth={1.25} aria-hidden />
                  <p className="max-w-md text-lg font-medium text-foreground">
                    {"What's on the agenda today?"}
                  </p>
                </div>
              ) : (
                <div className="space-y-6 pb-8">
                  {chatbotMessages.map((m) => {
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
                          <Button
                            variant="outline"
                            size="sm"
                            type="button"
                            className="text-xs"
                            onClick={() => {
                              const blob = new Blob([m.content], { type: 'text/markdown;charset=utf-8' });
                              const a = document.createElement('a');
                              a.href = URL.createObjectURL(blob);
                              a.download = 'chatbot-reply.md';
                              a.click();
                              URL.revokeObjectURL(a.href);
                            }}
                          >
                            <Download className="h-3.5 w-3.5" /> Download reply
                          </Button>
                        </div>
                      </div>
                    );
                  })}
                  {chatbotBusy ? (
                    <div className="animate-fade-in flex justify-start">
                      <div className="flex items-center gap-2 rounded-2xl border border-border/50 bg-secondary/40 px-4 py-3 text-sm text-muted-foreground">
                        <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
                        Thinking…
                      </div>
                    </div>
                  ) : null}
                  <div ref={chatbotChatEndRef} />
                </div>
              )}
            </div>
          </div>
          <div className="shrink-0 border-t border-border/50 bg-background/95 px-4 py-3 backdrop-blur-md sm:px-6">
            <div className="mx-auto w-full max-w-4xl space-y-3 lg:max-w-5xl xl:max-w-6xl">
              <div className="flex items-end gap-2 rounded-[1.75rem] border border-border/60 bg-secondary/25 p-2 shadow-sm focus-within:border-primary/30 focus-within:ring-2 focus-within:ring-primary/15">
                <div className="relative shrink-0">
                  <ServicePlusMenu
                    isOpen={plusMenuOpen}
                    onClose={() => setPlusMenuOpen(false)}
                    onSelectChatbot={selectChatbotFromMenu}
                    onSelectCapability={selectCapabilityFromMenu}
                    onSelectHybridRag={selectHybridRagFromMenu}
                    capabilities={capabilities}
                    capabilitiesDisabled={Boolean(capError)}
                    anchorRef={plusButtonRef}
                  />
                  <button
                    ref={plusButtonRef}
                    type="button"
                    title="Services menu"
                    onClick={() => setPlusMenuOpen((v) => !v)}
                    className="flex h-11 w-11 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
                  >
                    <Plus className="h-6 w-6 stroke-[2]" aria-hidden />
                    <span className="sr-only">Open services</span>
                  </button>
                </div>
                <textarea
                  rows={1}
                  value={chatbotInput}
                  onChange={(e) => setChatbotInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      if (!chatbotBusy && chatbotInput.trim()) handleChatbotSend();
                    }
                  }}
                  placeholder="Ask anything…"
                  disabled={chatbotBusy}
                  className="max-h-40 min-h-[44px] flex-1 resize-none bg-transparent py-3 pr-2 text-[15px] leading-snug text-foreground placeholder:text-muted-foreground focus:outline-none disabled:cursor-not-allowed disabled:opacity-50"
                />
                <Button
                  variant="warm"
                  size="icon"
                  className="h-11 w-11 shrink-0 rounded-full"
                  type="button"
                  onClick={handleChatbotSend}
                  disabled={chatbotBusy || !chatbotInput.trim()}
                >
                  {chatbotBusy ? <Loader2 className="h-5 w-5 animate-spin" /> : <Send className="h-5 w-5" />}
                </Button>
              </div>
              <p className="text-center text-[11px] text-muted-foreground">
                Enter sends · Shift+Enter newline · Local GGUF (OgesAssistant chatbot)
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="flex min-h-0 flex-1 flex-col">
          <div className="relative flex min-h-0 flex-1 flex-col overflow-y-auto">
            <div className="mx-auto flex min-h-full w-full max-w-none flex-1 flex-col px-4 py-4 sm:px-6 lg:px-10 xl:px-14 2xl:px-20">
              {docMessages.length === 0 && !selectedCap ? (
                <div className="flex flex-1 flex-col items-center justify-center pb-28 text-center">
                  <MessageSquare className="mb-3 h-10 w-10 text-muted-foreground/35" strokeWidth={1.25} aria-hidden />
                  <p className="max-w-lg text-sm text-muted-foreground">
                    Use <strong className="text-foreground">+</strong> to pick a document type,{' '}
                    <strong className="text-foreground">chatbot</strong> for Q&amp;A, or{' '}
                    <strong className="text-foreground">Hybrid RAG</strong> to chat over an uploaded file.
                  </p>
                </div>
              ) : (
                <div className="space-y-6 pb-4">
                  {docMessages.map((m) => {
                    if (m.role === 'user') {
                      return (
                        <div key={m.id} className="animate-fade-in flex w-full min-w-0 justify-end">
                          <div className="max-w-[min(100%,52rem)] rounded-3xl bg-primary/12 px-5 py-3 text-[15px] leading-relaxed text-foreground sm:px-6 sm:py-4">
                            {m.content}
                          </div>
                        </div>
                      );
                    }
                    if (m.role === 'assistant') {
                      return renderDocAssistantMessage(m);
                    }
                    return null;
                  })}
                  {documentFormCard}
                  <div ref={docChatEndRef} />
                </div>
              )}
            </div>
          </div>

          <div className="shrink-0 border-t border-border/50 bg-background/95 px-4 py-3 backdrop-blur-md sm:px-6 lg:px-10 xl:px-14 2xl:px-20">
            <div className="mx-auto w-full max-w-none">
              <div className="flex items-center gap-2 rounded-[1.75rem] border border-border/60 bg-secondary/25 p-2 shadow-sm">
                <div className="relative shrink-0">
                  <ServicePlusMenu
                    isOpen={plusMenuOpen}
                    onClose={() => setPlusMenuOpen(false)}
                    onSelectChatbot={selectChatbotFromMenu}
                    onSelectCapability={selectCapabilityFromMenu}
                    onSelectHybridRag={selectHybridRagFromMenu}
                    capabilities={capabilities}
                    capabilitiesDisabled={Boolean(capError)}
                    anchorRef={plusButtonRef}
                  />
                  <button
                    ref={plusButtonRef}
                    type="button"
                    title="Choose service"
                    onClick={() => setPlusMenuOpen((v) => !v)}
                    className="flex h-11 w-11 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
                  >
                    <Plus className="h-6 w-6 stroke-[2]" aria-hidden />
                    <span className="sr-only">Open services menu</span>
                  </button>
                </div>
                <p className="flex-1 py-3 text-sm text-muted-foreground">
                  {selectedCap
                    ? 'Complete the form above, or + to switch service.'
                    : '+ Document generation, chatbot, or Hybrid RAG'}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
