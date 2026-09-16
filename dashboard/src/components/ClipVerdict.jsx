import React, { useState, useEffect } from 'react';
import { ThumbsUp, ThumbsDown } from 'lucide-react';
import { apiFetch } from '../lib/api';

// Why a clip was not worth posting. Mirrors verdicts.REASONS on the server,
// which validates it — a value missing here is a 400, never a silent write.
const REASONS = [
    ['no_payoff', 'no payoff'],
    ['needs_context', 'needs context'],
    ['nothing_to_watch', 'nothing to watch'],
    ['boring', 'boring'],
    ['wrong_moment', 'wrong moment'],
    ['bad_cut', 'bad cut'],
];

// One fetch per job, not per card: a six-clip job would otherwise make six
// identical requests on mount. Keyed by job, cleared when a job is re-rated.
const cache = new Map();

async function loadJob(jobId) {
    if (!cache.has(jobId)) {
        cache.set(jobId, apiFetch(`/api/verdicts?job_id=${encodeURIComponent(jobId)}`)
            .then((r) => (r.ok ? r.json() : { verdicts: {} }))
            .catch(() => ({ verdicts: {} })));
    }
    return cache.get(jobId);
}

export default function ClipVerdict({ jobId, index }) {
    const [verdict, setVerdict] = useState(null);
    const [reason, setReason] = useState(null);
    const [asking, setAsking] = useState(false);
    const [saving, setSaving] = useState(false);

    useEffect(() => {
        let alive = true;
        if (!jobId || index === undefined) return undefined;
        loadJob(jobId).then((data) => {
            if (!alive) return;
            const mine = (data.verdicts || {})[String(index)] || (data.verdicts || {})[index];
            if (mine) { setVerdict(mine.verdict); setReason(mine.reason || null); }
        });
        return () => { alive = false; };
    }, [jobId, index]);

    const send = async (value, why = null) => {
        const prev = { verdict, reason };
        setSaving(true);
        // Optimistic, because the thumb is the whole interaction. But apiFetch
        // only throws on 402 — a 404 (the job aged out) comes back as a plain
        // response, so revert on anything that did not actually store.
        setVerdict(value);
        setReason(why);
        try {
            const res = await apiFetch('/api/verdict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    job_id: jobId, clip_index: index, verdict: value,
                    ...(why ? { reason: why } : {}),
                }),
            });
            if (!res.ok) { setVerdict(prev.verdict); setReason(prev.reason); }
            else { cache.delete(jobId); }
        } catch {
            setVerdict(prev.verdict);
            setReason(prev.reason);
        } finally {
            setSaving(false);
        }
    };

    const thumb = (value, Icon, label) => {
        const on = verdict === value;
        return (
            <button
                type="button"
                disabled={saving}
                aria-pressed={on}
                aria-label={label}
                title={label}
                onClick={() => {
                    // Clicking the thumb that is already on clears the rating.
                    // Without this there is no way back out of a misclick, and
                    // "no opinion" is a different thing from "bad".
                    if (on) { setAsking(false); send('unrated'); return; }
                    setAsking(value === 'bad');
                    send(value, null);
                }}
                className={`flex items-center gap-1 px-2 py-1 rounded-input border text-[11px] lowercase transition-colors disabled:opacity-45 ${
                    on ? 'border-brass text-brass' : 'border-rule text-muted hover:text-ink2'
                }`}
            >
                <Icon size={13} />
                {label}
            </button>
        );
    };

    return (
        <div className="mt-3 pt-3 border-t border-rule">
            <div className="flex items-center gap-2 flex-wrap">
                <span className="text-[11px] lowercase text-muted mr-auto">
                    worth posting?{verdict ? ' · click again to clear' : ''}
                </span>
                {thumb('good', ThumbsUp, 'yes')}
                {thumb('bad', ThumbsDown, 'no')}
            </div>

            {verdict === 'bad' && asking && (
                <div className="mt-2 flex flex-wrap gap-1.5">
                    {REASONS.map(([value, label]) => (
                        <button
                            key={value}
                            type="button"
                            onClick={() => { send('bad', value); setAsking(false); }}
                            className={`px-2 py-1 rounded-input border text-[11px] lowercase transition-colors ${
                                reason === value
                                    ? 'border-brass text-brass'
                                    : 'border-rule text-muted hover:text-ink2'
                            }`}
                        >
                            {label}
                        </button>
                    ))}
                </div>
            )}

            {verdict === 'bad' && !asking && reason && (
                <p className="mt-1.5 text-[11px] lowercase text-muted">
                    {REASONS.find(([v]) => v === reason)?.[1] || reason}
                    {' · '}
                    <button
                        type="button"
                        className="underline hover:text-ink2"
                        onClick={() => setAsking(true)}
                    >
                        change
                    </button>
                </p>
            )}
        </div>
    );
}
