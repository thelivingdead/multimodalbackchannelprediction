import { FormEvent, useMemo, useState } from "react";

type PredictResponse = {
  prediction: string;
  confidence: number;
  probabilities: Record<string, number>;
  cues: { name: string; score: number; detail: string }[];
  mode: string;
  classes: string[];
  modalities_used: string[];
  explanation: string;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

const EXAMPLES = [
  "Yeah, that makes sense — go on.",
  "No, I don't think I agree with that.",
  "Hmm, maybe… I'm not sure yet.",
  "Wow, really? That's interesting.",
];

export default function Predict() {
  const [text, setText] = useState(EXAMPLES[0]);
  const [useText, setUseText] = useState(true);
  const [useAudio, setUseAudio] = useState(false);
  const [useVideo, setUseVideo] = useState(false);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  const modalities = useMemo(() => {
    const m: string[] = [];
    if (useText) m.push("text");
    if (useAudio) m.push("audio");
    if (useVideo) m.push("video");
    return m.length ? m : ["text"];
  }, [useText, useAudio, useVideo]);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      let data: PredictResponse;
      if (audioFile || videoFile) {
        const form = new FormData();
        form.append("text", text);
        form.append("modalities", modalities.join(","));
        if (audioFile) form.append("audio", audioFile);
        if (videoFile) form.append("video", videoFile);
        const res = await fetch(`${API_BASE}/predict/upload`, { method: "POST", body: form });
        if (!res.ok) throw new Error(`API ${res.status}`);
        data = await res.json();
      } else {
        const res = await fetch(`${API_BASE}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, modalities, use_checkpoint: false }),
        });
        if (!res.ok) throw new Error(`API ${res.status}`);
        data = await res.json();
      }
      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? `${err.message}. Is the FastAPI server running on :8000?`
          : "Prediction failed"
      );
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  const sortedProbs = result
    ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <section className="predict">
      <div className="predict-intro">
        <h1>Predict</h1>
        <p>
          Paste listener/speaker context text, optionally attach audio or video. The demo API runs
          FLAME-style heuristic rules (swap in a trained checkpoint later).
        </p>
      </div>

      <form className="predict-form" onSubmit={onSubmit}>
        <label className="field">
          <span>Transcript / context</span>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={5}
            placeholder="e.g. Yeah, right — keep going."
          />
        </label>

        <div className="examples">
          {EXAMPLES.map((ex) => (
            <button key={ex} type="button" className="example" onClick={() => setText(ex)}>
              {ex}
            </button>
          ))}
        </div>

        <fieldset className="mods">
          <legend>Modalities</legend>
          <label>
            <input type="checkbox" checked={useText} onChange={(e) => setUseText(e.target.checked)} />
            Text
          </label>
          <label>
            <input
              type="checkbox"
              checked={useAudio}
              onChange={(e) => setUseAudio(e.target.checked)}
            />
            Audio
          </label>
          <label>
            <input
              type="checkbox"
              checked={useVideo}
              onChange={(e) => setUseVideo(e.target.checked)}
            />
            Video
          </label>
        </fieldset>

        <div className="uploads">
          <label className="field">
            <span>Audio file (optional)</span>
            <input
              type="file"
              accept="audio/*"
              onChange={(e) => setAudioFile(e.target.files?.[0] ?? null)}
            />
          </label>
          <label className="field">
            <span>Video file (optional)</span>
            <input
              type="file"
              accept="video/*"
              onChange={(e) => setVideoFile(e.target.files?.[0] ?? null)}
            />
          </label>
        </div>

        <button className="btn primary" type="submit" disabled={loading}>
          {loading ? "Predicting…" : "Predict backchannel class"}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {result && (
        <div className="results">
          <div className="result-hero">
            <p className="eyebrow">Top prediction</p>
            <h2>{result.prediction}</h2>
            <p className="conf">{(result.confidence * 100).toFixed(1)}% confidence · mode `{result.mode}`</p>
            <p className="explain">{result.explanation}</p>
          </div>
          <div className="bars">
            {sortedProbs.map(([cls, p]) => (
              <div key={cls} className="bar-row">
                <span className="bar-label">{cls}</span>
                <div className="bar-track">
                  <div className="bar-fill" style={{ width: `${Math.max(p * 100, 2)}%` }} />
                </div>
                <span className="bar-val">{(p * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
          {result.cues.length > 0 && (
            <ul className="cues">
              {result.cues.map((c) => (
                <li key={`${c.name}-${c.detail}`}>
                  <strong>{c.name}</strong> — {c.detail}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
