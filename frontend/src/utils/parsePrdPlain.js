/**
 * Parse backend PRD plain text: line 1 title, then ALL CAPS section headers, bullet lines "- ".
 * Returns { title, sections: [{ heading, bullets }] } or null if structure not detected.
 */
export function parsePrdPlain(text) {
  if (!text || typeof text !== 'string') return null;
  const lines = text.split(/\r?\n/);
  let i = 0;
  while (i < lines.length && !lines[i].trim()) i++;
  if (i >= lines.length) return null;
  const title = lines[i].trim();
  i += 1;
  while (i < lines.length && !lines[i].trim()) i++;

  const isSectionHeader = (line) => {
    const t = line.trim();
    if (!t || t.startsWith('-')) return false;
    if (t !== t.toUpperCase()) return false;
    if (t.length < 3 || t.length > 120) return false;
    if (/^[-_=•.]+$/.test(t)) return false;
    return /^[A-Z0-9 &,.()/-]+$/.test(t);
  };

  const sections = [];
  let current = null;

  for (; i < lines.length; i++) {
    const raw = lines[i];
    const trimmed = raw.trim();
    if (!trimmed) continue;

    if (isSectionHeader(raw)) {
      current = { heading: trimmed, bullets: [] };
      sections.push(current);
      continue;
    }

    const bulletMatch = raw.match(/^(\s*)-\s+(.*)$/);
    if (bulletMatch) {
      const lead = bulletMatch[1];
      const bullet = bulletMatch[2].trim();
      const depth = Math.min(4, Math.floor(lead.length / 2));
      const item = { text: bullet, depth };
      if (current) {
        current.bullets.push(item);
      } else {
        if (!sections.length) {
          current = { heading: 'OVERVIEW', bullets: [] };
          sections.push(current);
        }
        sections[sections.length - 1].bullets.push(item);
      }
    }
  }

  if (sections.length === 0) return null;
  return { title, sections };
}

/** Title-case a section heading for display (keeps acronyms-ish). */
export function formatSectionHeading(heading) {
  return heading
    .toLowerCase()
    .split(' ')
    .map((w) => (w.length ? w[0].toUpperCase() + w.slice(1) : w))
    .join(' ');
}
