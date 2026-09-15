import React, { useState, useEffect, useRef } from 'react';
import { Link2, Upload, FileVideo, X, Info, Loader2, ChevronDown } from 'lucide-react';
import { getApiUrl } from '../config';

const SUPPORTED_PLATFORMS = [
    'YouTube', 'Vimeo', 'TikTok', 'X / Twitter', 'Twitch',
    'Facebook', 'Instagram', 'Dailymotion', 'Reddit', 'Streamable',
];

// Mirrors clip_selection.CLIP_INSTRUCTIONS_MAX_CHARS; the API rejects longer text.
const CLIP_INSTRUCTIONS_MAX = 1000;
// Mirrors transcript_import.TRANSCRIPT_MAX_CHARS; the API rejects longer text.
const TRANSCRIPT_MAX = 300000;

// Starters for the instructions box. Each one INSERTS editable text: nothing is
// hidden, what is in the box is exactly what the AI receives.
const INSTRUCTION_PRESETS = [
    { label: 'funny', text: 'Pick the funniest moments: jokes, banter, reactions and absurd situations.' },
    { label: 'insights & tips', text: 'Pick moments that teach something: a concrete tip, a surprising fact or a clear takeaway.' },
    { label: 'stories', text: 'Pick emotional or personal stories that have a clear payoff.' },
    { label: 'hot takes', text: 'Pick bold opinions, disagreements and controversial statements.' },
    { label: 'skip filler', text: 'Never pick intros, outros, sponsor reads, ads or housekeeping.' },
];

export default function MediaInput({ onProcess, isProcessing }) {
    const [youtubeUrlEnabled, setYoutubeUrlEnabled] = useState(true);
    // File upload is the primary path; the link is secondary.
    const [mode, setMode] = useState('file'); // 'file' | 'url'
    const [url, setUrl] = useState('');
    const [file, setFile] = useState(null);
    const [acknowledged, setAcknowledged] = useState(false);
    const [outputFormat, setOutputFormat] = useState('vertical'); // vertical | horizontal | square
    const [showInfo, setShowInfo] = useState(false);
    // Advanced generation controls — empty string means "let the AI decide",
    // which keeps the default pipeline behavior untouched.
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [targetClips, setTargetClips] = useState('');
    const [clipMinSeconds, setClipMinSeconds] = useState('');
    const [clipMaxSeconds, setClipMaxSeconds] = useState('');
    // Auto-hook: burn the AI hook text into every clip. On by default; the
    // choice persists so turning it off sticks across sessions.
    const [autoHook, setAutoHook] = useState(() => {
        try { return localStorage.getItem('os_auto_hook') !== '0'; } catch { return true; }
    });
    const [autoHookStyle, setAutoHookStyle] = useState(() => {
        try { return localStorage.getItem('os_auto_hook_style') || 'classic'; } catch { return 'classic'; }
    });
    // Layout: 'auto' lets the AI pick per video (server default); the others
    // force one on so a podcast host who knows what they uploaded doesn't
    // depend on the detector, and 'none' keeps the plain single crop.
    const [layout, setLayout] = useState(() => {
        try { return localStorage.getItem('os_layout') || 'auto'; } catch { return 'auto'; }
    });
    // Download quality for pasted links ("up to"; a video that tops out lower
    // gets its best). Uploads keep their own resolution, so the row is hidden
    // there. Persisted like the layout so a chosen quality sticks.
    const [quality, setQuality] = useState(() => {
        try { return localStorage.getItem('os_quality') || '1080'; } catch { return '1080'; }
    });
    // Creator instructions for clip selection. Deliberately NOT persisted: they
    // are usually about one video, and a remembered "only moments about pricing"
    // would quietly narrow the next, unrelated one. Presets cover recurring styles.
    const [clipInstructions, setClipInstructions] = useState('');
    // A transcript the user already has (YouTube's transcript panel, SRT/VTT):
    // the job then never transcribes the whole video. Per video, never persisted.
    const [useTranscript, setUseTranscript] = useState(false);
    const [transcript, setTranscript] = useState('');
    const transcriptBlocked = useTranscript && (!transcript.trim() || transcript.length > TRANSCRIPT_MAX);
    const loadTranscriptFile = (e) => {
        const picked = e.target.files?.[0];
        e.target.value = '';
        if (picked) picked.text().then(setTranscript).catch(() => {});
    };
    const addInstructionPreset = (text) => {
        setClipInstructions((current) => {
            if (current.includes(text)) return current;
            const next = current.trim() ? `${current.trimEnd()}\n${text}` : text;
            // Never append half a sentence: skip the preset if it doesn't fit.
            return next.length > CLIP_INSTRUCTIONS_MAX ? current : next;
        });
    };
    const infoRef = useRef(null);

    // Close the compatibility popover on any outside click.
    useEffect(() => {
        if (!showInfo) return;
        const onClick = (e) => {
            if (infoRef.current && !infoRef.current.contains(e.target)) setShowInfo(false);
        };
        document.addEventListener('mousedown', onClick);
        return () => document.removeEventListener('mousedown', onClick);
    }, [showInfo]);

    useEffect(() => {
        fetch(getApiUrl('/api/config'))
            .then((r) => r.ok ? r.json() : null)
            .then((cfg) => {
                if (cfg && cfg.youtubeUrlEnabled === false) {
                    setYoutubeUrlEnabled(false);
                    setMode('file');
                }
            })
            .catch(() => {});
    }, []);

    // A link pasted in the landing hero: preload it here so the user picks up
    // where they left off. Not auto-submitted — the rights attestation below
    // has to be ticked by the user.
    useEffect(() => {
        let pending = null;
        try {
            pending = localStorage.getItem('os_pending_url');
            if (pending) localStorage.removeItem('os_pending_url');
        } catch { /* ignore */ }
        if (pending) {
            setMode('url');
            setUrl(pending);
        }
    }, []);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!acknowledged || transcriptBlocked) return;
        const advanced = {
            targetClips: targetClips || null,
            clipMinSeconds: clipMinSeconds || null,
            clipMaxSeconds: clipMaxSeconds || null,
            autoHook,
            autoHookStyle,
            layout,
            clipInstructions: clipInstructions.trim() || null,
            transcript: useTranscript && transcript.trim() ? transcript : null,
        };
        try {
            localStorage.setItem('os_auto_hook', autoHook ? '1' : '0');
            localStorage.setItem('os_auto_hook_style', autoHookStyle);
            localStorage.setItem('os_layout', layout);
            localStorage.setItem('os_quality', quality);
        } catch { /* ignore */ }
        if (mode === 'url' && url) {
            onProcess({ type: 'url', payload: url, acknowledged: true, outputFormat, quality, ...advanced });
        } else if (mode === 'file' && file) {
            onProcess({ type: 'file', payload: file, acknowledged: true, outputFormat, ...advanced });
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0]);
            setMode('file');
        }
    };

    return (
        <div className="card p-4 sm:p-6 animate-fade">
            <div className="flex gap-4 sm:gap-6 mb-6 border-b border-rule">
                <button
                    onClick={() => setMode('file')}
                    className={`flex items-center gap-2 pb-3 px-1 -mb-px border-b-2 text-sm lowercase whitespace-nowrap transition-colors ${mode === 'file'
                        ? 'text-ink border-brass'
                        : 'text-muted border-transparent hover:text-ink2'
                        }`}
                >
                    <Upload size={16} className={`hidden sm:block ${mode === 'file' ? 'text-brass' : ''}`} />
                    Upload File
                </button>
                {youtubeUrlEnabled && (
                    <button
                        onClick={() => setMode('url')}
                        className={`flex items-center gap-2 pb-3 px-1 -mb-px border-b-2 text-sm lowercase whitespace-nowrap transition-colors ${mode === 'url'
                            ? 'text-ink border-brass'
                            : 'text-muted border-transparent hover:text-ink2'
                            }`}
                    >
                        <Link2 size={16} className={`hidden sm:block ${mode === 'url' ? 'text-brass' : ''}`} />
                        Video URL
                    </button>
                )}
            </div>

            <form onSubmit={handleSubmit}>
                {mode === 'url' ? (
                    <div className="space-y-4">
                        <div className="relative">
                            <input
                                type="url"
                                value={url}
                                onChange={(e) => setUrl(e.target.value)}
                                placeholder="https://... paste a video link"
                                className="input-field pr-11"
                                required
                            />
                            <div className="absolute inset-y-0 right-2 flex items-center" ref={infoRef}>
                                <button
                                    type="button"
                                    onClick={() => setShowInfo((v) => !v)}
                                    aria-label="Supported platforms"
                                    className="p-1.5 text-muted hover:text-brass transition-colors"
                                >
                                    <Info size={16} />
                                </button>
                                {showInfo && (
                                    <div className="absolute right-0 top-full mt-2 w-64 z-20 card p-4 text-left animate-fade">
                                        <p className="eyebrow mb-2">Paste a link from</p>
                                        <div className="flex flex-wrap gap-1.5">
                                            {SUPPORTED_PLATFORMS.map((p) => (
                                                <span key={p} className="text-xs px-2 py-0.5 rounded-full bg-paper3 text-ink2">
                                                    {p}
                                                </span>
                                            ))}
                                        </div>
                                        <p className="text-xs text-muted mt-2.5 leading-relaxed">
                                            …and 1,000+ more sites. If a link has a public video, we can usually fetch it.
                                        </p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                ) : (
                    <div
                        className={`border-2 border-dashed rounded-card p-6 sm:p-8 text-center transition-colors ${file ? 'border-brass' : 'border-rule2 hover:border-brass'
                            }`}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={handleDrop}
                    >
                        {file ? (
                            <div className="flex items-center justify-center gap-3 text-ok min-w-0">
                                <FileVideo size={18} className="shrink-0" />
                                <span className="font-medium truncate">{file.name}</span>
                                <button
                                    type="button"
                                    onClick={() => setFile(null)}
                                    className="p-1 text-muted hover:text-ink hover:bg-paper3 rounded-full transition-colors"
                                >
                                    <X size={16} />
                                </button>
                            </div>
                        ) : (
                            <label className="cursor-pointer block">
                                <input
                                    type="file"
                                    accept="video/*"
                                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                                    className="hidden"
                                />
                                <Upload className="mx-auto mb-3 text-muted" size={18} />
                                <p className="text-ink2 lowercase">Click to upload or drag and drop</p>
                                <p className="readout mt-2">MP4, MOV up to 500MB</p>
                            </label>
                        )}
                    </div>
                )}

                {/* Output format selector */}
                <div className="mt-5">
                    <p className="eyebrow mb-2">Output format</p>
                    <div className="grid grid-cols-3 gap-2">
                        {[
                            { value: 'vertical', label: '9:16', hint: 'Shorts · Reels · TikTok', w: 18, h: 32 },
                            { value: 'square', label: '1:1', hint: 'Feed posts', w: 28, h: 28 },
                            { value: 'horizontal', label: '16:9', hint: 'Keep landscape · YouTube', w: 36, h: 20 },
                        ].map((f) => {
                            const active = outputFormat === f.value;
                            return (
                                <button
                                    key={f.value}
                                    type="button"
                                    onClick={() => setOutputFormat(f.value)}
                                    className={`py-3 px-2 rounded-input border flex flex-col items-center gap-2 transition-colors
                                        ${active ? 'border-[color:var(--color-accent)] text-ink' : 'border-rule2 text-muted hover:border-[color:var(--color-accent)]'}`}
                                >
                                    {/* Aspect-ratio glyph */}
                                    <span
                                        className="rounded-[3px] border-2 transition-colors"
                                        style={{
                                            width: `${f.w}px`,
                                            height: `${f.h}px`,
                                            borderColor: active ? 'var(--color-accent)' : 'var(--color-rule-2)',
                                            backgroundColor: active ? 'color-mix(in srgb, var(--color-accent) 22%, transparent)' : 'transparent',
                                        }}
                                    />
                                    <span className="block font-mono text-sm leading-none">{f.label}</span>
                                    <span className="block text-[11px] sm:text-[10px] leading-tight text-center text-muted">{f.hint}</span>
                                </button>
                            );
                        })}
                    </div>
                </div>

                {/* Creator instructions — optional; steers every clip-selection stage */}
                <div className="mt-5">
                    <div className="flex items-baseline justify-between gap-3 mb-2">
                        <p className="eyebrow">what to clip · optional</p>
                        <span className="readout">{clipInstructions.length}/{CLIP_INSTRUCTIONS_MAX}</span>
                    </div>
                    <textarea
                        value={clipInstructions}
                        onChange={(e) => setClipInstructions(e.target.value)}
                        rows={2}
                        maxLength={CLIP_INSTRUCTIONS_MAX}
                        className="input-field resize-none text-sm"
                        placeholder="e.g. only moments where the guest gives a concrete tip; skip the sponsor read"
                        aria-label="clip instructions"
                    />
                    <div className="mt-2 flex flex-wrap gap-1.5">
                        {INSTRUCTION_PRESETS.map((preset) => (
                            <button
                                key={preset.label}
                                type="button"
                                onClick={() => addInstructionPreset(preset.text)}
                                className="btn-quiet !px-2.5 !py-1 !text-xs"
                                title={preset.text}
                            >
                                + {preset.label}
                            </button>
                        ))}
                    </div>
                </div>

                {/* A transcript the user already has — optional; skips transcribing the whole video */}
                <div className="mt-4">
                    <label className="flex items-center gap-2 text-xs text-ink2 cursor-pointer select-none">
                        <input
                            type="checkbox"
                            checked={useTranscript}
                            onChange={(e) => setUseTranscript(e.target.checked)}
                            className="w-4 h-4 shrink-0 accent-[var(--color-accent)] cursor-pointer"
                        />
                        I have the transcript · skip transcribing the whole video
                    </label>
                    {useTranscript && (
                        <div className="mt-2 animate-fade">
                            <textarea
                                value={transcript}
                                onChange={(e) => setTranscript(e.target.value)}
                                rows={6}
                                className="input-field text-xs font-mono"
                                placeholder={'0:00\nso today I want to talk about…\n0:04\nwhy most people get it wrong\n\nor an SRT / VTT caption file'}
                                aria-label="transcript"
                            />
                            <div className="mt-1.5 flex items-start justify-between gap-3">
                                <p className="text-left text-[11px] leading-relaxed text-muted">
                                    YouTube: description → Show transcript → copy the list with its times.
                                    SRT and VTT caption files work too. The transcript chooses the moments;
                                    only the chosen clips get transcribed, for exact cuts and captions.
                                </p>
                                <span className={`readout shrink-0 ${transcript.length > TRANSCRIPT_MAX ? 'text-danger' : ''}`}>
                                    {transcript.length.toLocaleString()}/{TRANSCRIPT_MAX.toLocaleString()}
                                </span>
                            </div>
                            <label className="btn-quiet !px-2.5 !py-1 !text-xs mt-2 inline-flex cursor-pointer">
                                load a caption file
                                <input
                                    type="file"
                                    accept=".srt,.vtt,.txt,.json"
                                    onChange={loadTranscriptFile}
                                    className="hidden"
                                />
                            </label>
                        </div>
                    )}
                </div>

                {/* Advanced generation controls — collapsed by default; blank = AI decides */}
                <div className="mt-4">
                    <button
                        type="button"
                        onClick={() => setShowAdvanced((v) => !v)}
                        className="flex items-center gap-1.5 text-xs text-muted hover:text-ink2 lowercase transition-colors"
                    >
                        <ChevronDown size={14} className={`transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
                        advanced options
                        {(targetClips || clipMinSeconds || clipMaxSeconds || !autoHook
                            || (mode === 'url' && quality !== '1080')) && (
                            <span className="text-brass">·</span>
                        )}
                    </button>
                    {showAdvanced && (
                        /* Stacked on a phone: three number fields side by side leaves
                           ~100px each, which crushes both label and value. */
                        <div className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-2 animate-fade">
                            <div>
                                <p className="eyebrow mb-1.5">clips to aim for</p>
                                <input
                                    type="number" min="1" max="15" step="1"
                                    value={targetClips}
                                    onChange={(e) => setTargetClips(e.target.value)}
                                    placeholder="auto"
                                    className="input-field"
                                />
                            </div>
                            <div>
                                <p className="eyebrow mb-1.5">min length (s)</p>
                                <input
                                    type="number" min="5" max="175" step="1"
                                    value={clipMinSeconds}
                                    onChange={(e) => setClipMinSeconds(e.target.value)}
                                    placeholder="15"
                                    className="input-field"
                                />
                            </div>
                            <div>
                                <p className="eyebrow mb-1.5">max length (s)</p>
                                <input
                                    type="number" min="10" max="180" step="1"
                                    value={clipMaxSeconds}
                                    onChange={(e) => setClipMaxSeconds(e.target.value)}
                                    placeholder="60"
                                    className="input-field"
                                />
                            </div>
                            <p className="col-span-1 sm:col-span-3 text-[11px] leading-relaxed text-muted">
                                Targets, not guarantees: the AI returns fewer clips when the
                                material doesn't hold them. Leave blank to let it decide.
                            </p>
                            <div className="col-span-1 sm:col-span-3 flex flex-wrap items-center justify-between gap-3 pt-3 sm:pt-1 border-t border-rule">
                                <span className="text-xs text-ink2">vertical layout</span>
                                <select
                                    value={layout}
                                    onChange={(e) => setLayout(e.target.value)}
                                    className="input-field !w-auto text-xs py-1.5"
                                    aria-label="vertical layout"
                                >
                                    <option value="auto">Auto (AI picks per video)</option>
                                    <option value="split">Two speakers stacked</option>
                                    <option value="screencast">Screen over presenter</option>
                                    <option value="none">Single crop only</option>
                                </select>
                            </div>
                            {mode === 'url' && (
                                <div className="col-span-1 sm:col-span-3 flex flex-wrap items-center justify-between gap-3 pt-3 sm:pt-1 border-t border-rule">
                                    <span className="text-xs text-ink2">download quality</span>
                                    <select
                                        value={quality}
                                        onChange={(e) => setQuality(e.target.value)}
                                        className="input-field !w-auto text-xs py-1.5"
                                        aria-label="download quality"
                                    >
                                        <option value="360">360p</option>
                                        <option value="480">480p</option>
                                        <option value="720">720p</option>
                                        <option value="1080">1080p (recommended)</option>
                                        <option value="1440">1440p (large download)</option>
                                        <option value="2160">4K (very large, slow)</option>
                                    </select>
                                </div>
                            )}
                            <div className="col-span-1 sm:col-span-3 flex flex-wrap items-center justify-between gap-3 pt-3 sm:pt-1 border-t border-rule">
                                <label className="flex items-center gap-2 text-xs text-ink2 cursor-pointer select-none">
                                    <input
                                        type="checkbox"
                                        checked={autoHook}
                                        onChange={(e) => setAutoHook(e.target.checked)}
                                        className="w-4 h-4 shrink-0 accent-[var(--color-accent)] cursor-pointer"
                                    />
                                    auto hook titles on clips
                                </label>
                                {autoHook && (
                                    <select
                                        value={autoHookStyle}
                                        onChange={(e) => setAutoHookStyle(e.target.value)}
                                        className="input-field !w-auto text-xs py-1.5"
                                    >
                                        <option value="classic">Classic</option>
                                        <option value="dark">Dark</option>
                                        <option value="yellow">Yellow</option>
                                        <option value="red">Red</option>
                                        <option value="outline">Outline</option>
                                        <option value="outline_yellow">Outline+</option>
                                    </select>
                                )}
                            </div>
                        </div>
                    )}
                </div>

                <label className="flex items-start gap-2.5 mt-5 text-left text-[13px] sm:text-xs leading-relaxed text-muted cursor-pointer select-none">
                    <input
                        type="checkbox"
                        checked={acknowledged}
                        onChange={(e) => setAcknowledged(e.target.checked)}
                        className="mt-0.5 w-4 h-4 shrink-0 accent-[var(--color-accent)] cursor-pointer"
                    />
                    <span>
                        I confirm I own this content or have the rights to process it. I am responsible for any content I submit. See our <a href="/terms" target="_blank" rel="noopener noreferrer" className="text-ink2 underline underline-offset-2 hover:text-brass transition-colors" onClick={(e) => e.stopPropagation()}>Terms</a> and <a href="/privacy" target="_blank" rel="noopener noreferrer" className="text-ink2 underline underline-offset-2 hover:text-brass transition-colors" onClick={(e) => e.stopPropagation()}>Privacy Policy</a>.
                    </span>
                </label>

                <button
                    type="submit"
                    disabled={isProcessing || !acknowledged || transcriptBlocked
                        || (mode === 'url' && !url) || (mode === 'file' && !file)}
                    className="w-full btn-primary mt-4"
                >
                    {isProcessing ? (
                        <>
                            <Loader2 size={16} className="animate-spin" />
                            Processing Video...
                        </>
                    ) : (
                        <>
                            Generate Clips
                        </>
                    )}
                </button>
            </form>
        </div>
    );
}
