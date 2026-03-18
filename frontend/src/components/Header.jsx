import React from 'react';

export default function Header({ activeTab, onTabChange }) {
  const NAV_ITEMS = [
    { id: 1, label: 'Document Summarization', dot: 'green' },
    { id: 2, label: 'Conversational AI', dot: 'green' },
    { id: 3, label: 'Voice-Based Form Filling', dot: 'blue' },
    { id: 4, label: 'Document OCR & Form Processing', dot: 'red' },
    { id: 5, label: 'Image Matching & Liveness', dot: 'red' },
  ];

  return (
    <>
      {/* --- Top Header --- */}
      <header className="app-header" id="app-header">
        <div className="header-left">
          <div className="header-emblem">🏛️</div>
          <div className="header-title">
            <h1>भारत सरकार - एआई सेवा मंच</h1>
            <p>Government of India — AI Services Platform</p>
          </div>
        </div>
        <div className="header-right">
          <span>Ministry of Electronics &<br />Information Technology</span>
          <button className="theme-toggle" aria-label="Toggle theme">🌙</button>
        </div>
      </header>

      {/* --- Navigation Bar --- */}
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
    </>
  );
}
