import React, { useEffect, useRef } from 'react';
import {
  FileText,
  Code2,
  GitBranch,
  FileStack,
  Shield,
  LayoutTemplate,
  Network,
  Search,
} from 'lucide-react';

const iconClass = 'h-5 w-5 shrink-0 text-zinc-400';

function MenuIcon({ id }) {
  switch (id) {
    case 'code_summarizer':
      return <Code2 className={iconClass} aria-hidden />;
    case 'tech_docs':
      return <FileStack className={iconClass} aria-hidden />;
    case 'flow_diagram':
      return <GitBranch className={iconClass} aria-hidden />;
    case 'code_review':
      return <Shield className={iconClass} aria-hidden />;
    case 'architecture':
      return <LayoutTemplate className={iconClass} aria-hidden />;
    case 'cdg':
      return <Network className={iconClass} aria-hidden />;
    default:
      return <FileText className={iconClass} aria-hidden />;
  }
}

/**
 * Dark vertical menu (reference: icon left, label right, hover row).
 * Positioned above the anchor via absolute bottom-full.
 */
export default function ServicePlusMenu({
  isOpen,
  onClose,
  onSelectCapability,
  onSelectHybridRag,
  capabilities,
  capabilitiesDisabled,
  anchorRef,
}) {
  const menuRef = useRef(null);

  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e) => {
      if (e.key === 'Escape') onClose();
    };
    const onMouseDown = (e) => {
      if (menuRef.current?.contains(e.target)) return;
      if (anchorRef?.current?.contains(e.target)) return;
      onClose();
    };
    document.addEventListener('keydown', onKey);
    document.addEventListener('mousedown', onMouseDown);
    return () => {
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('mousedown', onMouseDown);
    };
  }, [isOpen, onClose, anchorRef]);

  if (!isOpen) return null;

  return (
    <div
      ref={menuRef}
      role="menu"
      aria-label="Choose a service"
      className="absolute bottom-full left-0 z-50 mb-2 w-[min(100vw-1.5rem,18rem)] rounded-2xl border border-zinc-700/90 bg-zinc-900 py-1.5 shadow-2xl ring-1 ring-black/20"
    >
      {(capabilities || []).map((c) => (
        <button
          key={c.id}
          type="button"
          role="menuitem"
          disabled={capabilitiesDisabled}
          onClick={() => {
            if (capabilitiesDisabled) return;
            onSelectCapability(c.id);
          }}
          className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-sm text-zinc-100 transition-colors hover:bg-zinc-800 disabled:pointer-events-none disabled:opacity-40"
        >
          <MenuIcon id={c.id} />
          <span className="min-w-0 flex-1 leading-snug">{c.title}</span>
        </button>
      ))}
      {(capabilities || []).length > 0 ? <div className="my-1.5 h-px bg-zinc-700/80" role="separator" /> : null}
      <button
        type="button"
        role="menuitem"
        onClick={onSelectHybridRag}
        className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-sm text-zinc-100 transition-colors hover:bg-zinc-800"
      >
        <Search className={iconClass} aria-hidden />
        <span className="leading-snug">Hybrid RAG</span>
      </button>
    </div>
  );
}
