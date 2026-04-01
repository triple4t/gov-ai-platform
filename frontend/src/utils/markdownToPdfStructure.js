/**
 * Parse GitHub-style markdown into a flat sequence for jsPDF export.
 * @returns {{ title: string, parts: Array<{ type: 'h2'|'h3'|'p'|'bullet'|'code', text: string }> } | null}
 */
export function parseMarkdownForPdf(markdown) {
  if (!markdown || typeof markdown !== 'string') return null;
  const lines = markdown.split(/\r?\n/);
  const parts = [];
  let title = '';
  let paraBuf = [];

  const flushPara = () => {
    const t = paraBuf.join(' ').replace(/\s+/g, ' ').trim();
    paraBuf = [];
    if (t) parts.push({ type: 'p', text: t });
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const t = line.trimEnd();
    const trimmed = t.trim();

    if (!trimmed) {
      flushPara();
      continue;
    }

    if (trimmed.startsWith('```')) {
      flushPara();
      const fence = [trimmed];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith('```')) {
        fence.push(lines[i]);
        i += 1;
      }
      if (i < lines.length) fence.push(lines[i]);
      parts.push({ type: 'code', text: fence.join('\n') });
      continue;
    }

    if (trimmed.startsWith('# ') && !title) {
      flushPara();
      title = trimmed.slice(2).trim();
      continue;
    }
    if (trimmed.startsWith('## ')) {
      flushPara();
      parts.push({ type: 'h2', text: trimmed.slice(3).trim() });
      continue;
    }
    if (trimmed.startsWith('### ')) {
      flushPara();
      parts.push({ type: 'h3', text: trimmed.slice(4).trim() });
      continue;
    }
    if (/^[-*]\s+/.test(trimmed)) {
      flushPara();
      parts.push({ type: 'bullet', text: trimmed.replace(/^[-*]\s+/, '').trim() });
      continue;
    }
    if (/^\d+\.\s+/.test(trimmed)) {
      flushPara();
      parts.push({ type: 'bullet', text: trimmed.replace(/^\d+\.\s+/, '').trim() });
      continue;
    }
    if (trimmed.startsWith('>')) {
      flushPara();
      parts.push({ type: 'p', text: trimmed.replace(/^>\s?/, '').trim() });
      continue;
    }

    paraBuf.push(trimmed);
  }
  flushPara();

  if (!title) {
    const h2 = parts.find((p) => p.type === 'h2');
    title = h2 ? h2.text : 'Document';
  }
  return { title, parts };
}
