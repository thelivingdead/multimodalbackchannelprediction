import { Link } from "react-router-dom";

const CLASSES = [
  "nod",
  "shake",
  "tilt",
  "lean_forward",
  "lean_back",
  "eyebrow_raise",
  "neutral",
];

export default function Home() {
  return (
    <section className="hero">
      <div className="hero-copy">
        <p className="eyebrow">University of Surrey · CVSSP-style research demo</p>
        <h1>BackchannelAI</h1>
        <p className="lede">
          Predict fine-grained listener backchannel behaviours from text, audio, and visual
          cues—beyond binary keep / turn / backchannel labels.
        </p>
        <div className="cta-row">
          <Link className="btn primary" to="/predict">
            Run a prediction
          </Link>
          <Link className="btn ghost" to="/docs">
            Read the methodology
          </Link>
        </div>
      </div>
      <div className="hero-panel" aria-hidden="true">
        <div className="orbit">
          {CLASSES.map((c) => (
            <span key={c} className={`chip chip-${c}`}>
              {c}
            </span>
          ))}
        </div>
        <p className="panel-note">7-class taxonomy derived from FLAME / EMOCA head dynamics on Columbia RealTalk</p>
      </div>

      <div className="feature-strip">
        <article>
          <h2>Problem</h2>
          <p>
            Full-duplex dialogue needs typed listener feedback—nods, tilts, leans—not just a single
            “backchannel” bit.
          </p>
        </article>
        <article>
          <h2>Data</h2>
          <p>
            Columbia RealTalk dyadic video with FLAME parameters; MM-F2F as the tri-modal coarse
            baseline in the literature.
          </p>
        </article>
        <article>
          <h2>Method</h2>
          <p>
            Rule-derived labels, then multimodal fusion (text · audio · video · FLAME) toward a
            7-class softmax head.
          </p>
        </article>
      </div>
    </section>
  );
}
