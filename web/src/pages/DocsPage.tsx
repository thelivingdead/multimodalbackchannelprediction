export default function DocsPage() {
  return (
    <section className="docs">
      <h1>Docs & dissertation artefacts</h1>
      <p>
        These markdown reports live in the project <code>docs/</code> folder. Open them in your
        editor or paste into Word/LaTeX for the dissertation.
      </p>

      <div className="doc-grid">
        <article>
          <h2>Data analysis report</h2>
          <p>
            Problem framing, Columbia RealTalk vs MM-F2F, 7-class taxonomy, EDA plan, label
            derivation, and risks.
          </p>
          <p className="path">docs/01_data_analysis_report.md</p>
        </article>
        <article>
          <h2>Methodology & roadmap</h2>
          <p>
            Research questions, labelling protocol, encoder/fusion plan, baselines, metrics, and a
            10-week execution roadmap.
          </p>
          <p className="path">docs/02_research_methodology_and_roadmap.md</p>
        </article>
        <article>
          <h2>EDA notebook</h2>
          <p>
            Skeleton loaders, nod band-pass detector, and class histogram plots. Set{" "}
            <code>REALTALK_ROOT</code> when data is mounted.
          </p>
          <p className="path">notebooks/01_eda_skeleton.ipynb</p>
        </article>
        <article>
          <h2>Prediction API</h2>
          <p>
            FastAPI heuristic 7-class service with <code>/predict</code> and upload endpoint.
            Checkpoint hook ready for a future PyTorch model.
          </p>
          <p className="path">api/main.py · http://127.0.0.1:8000/docs</p>
        </article>
      </div>

      <h2>Class glossary</h2>
      <ul className="glossary">
        <li>
          <strong>nod</strong> — affirmation / continued attention via pitch oscillation
        </li>
        <li>
          <strong>shake</strong> — negation via yaw oscillation
        </li>
        <li>
          <strong>tilt</strong> — uncertainty / curiosity via roll
        </li>
        <li>
          <strong>lean_forward / lean_back</strong> — engagement vs distancing
        </li>
        <li>
          <strong>eyebrow_raise</strong> — surprise / social accent
        </li>
        <li>
          <strong>neutral</strong> — no event in the analysis window
        </li>
      </ul>
    </section>
  );
}
