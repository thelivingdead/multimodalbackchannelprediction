import { NavLink, Route, Routes } from "react-router-dom";
import Home from "./pages/Home";
import Predict from "./pages/Predict";
import DocsPage from "./pages/DocsPage";

export default function App() {
  return (
    <div className="shell">
      <header className="topnav">
        <NavLink to="/" className="brand">
          BackchannelAI
        </NavLink>
        <nav>
          <NavLink to="/" end>
            Overview
          </NavLink>
          <NavLink to="/predict">Predict</NavLink>
          <NavLink to="/docs">Docs</NavLink>
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/docs" element={<DocsPage />} />
        </Routes>
      </main>
      <footer className="footer">
        MSc dissertation demo · fine-grained multimodal backchannel prediction · heuristic API until checkpoint swap-in
      </footer>
    </div>
  );
}
