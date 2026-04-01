import React from 'react';

/**
 * PRD-only header (no tab strip). Restore `NAV_ITEMS` + props below for full platform.
 */
export default function Header(/* { activeTab, onTabChange } */) {
  // const NAV_ITEMS = [
  //   { id: 1, label: 'Document Summarization', dot: 'green' },
  //   { id: 2, label: 'Conversational AI', dot: 'green' },
  //   { id: 3, label: 'Voice-Based Form Filling', dot: 'blue' },
  //   { id: 4, label: 'Document OCR & Form Processing', dot: 'red' },
  //   { id: 5, label: 'Image Matching & Liveness', dot: 'red' },
  //   { id: 6, label: 'AI-Based PRD Platform', dot: 'green' },
  // ];

  return (
    <>
      <header className="app-header" id="app-header">
        <div className="header-left">
          <div className="header-emblem">📋</div>
          <div className="header-title">
            <h1>PRD Platform</h1>
            <p>Codebase upload, RAG, and document generation</p>
          </div>
        </div>
        <div className="header-right">
          <span>Government of India — AI Services Platform</span>
        </div>
      </header>

      {/* Full nav (pass activeTab, onTabChange from App when restoring tabs):
      <nav className="nav-bar" id="nav-bar">
        {NAV_ITEMS.map((item) => (
          <div
            key={item.id}
            className={`nav-item${activeTab === item.id ? ' active' : ''}`}
            id={`nav-item-${item.id}`}
            onClick={() => onTabChange(item.id)}
          >
            <span className={`nav-dot ${item.dot}`}></span>
            {item.id}. {item.label}
          </div>
        ))}
      </nav>
      */}
    </>
  );
}
