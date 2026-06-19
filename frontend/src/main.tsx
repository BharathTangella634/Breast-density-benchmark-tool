import React, { FormEvent, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Award,
  BarChart3,
  ClipboardCheck,
  Database,
  Gauge,
  History,
  Images,
  Trophy,
  Upload,
} from "lucide-react";
import "./styles.css";

type EvaluationResponse = {
  run: {
    id: number;
    created_at: string;
    submission_type: string;
    source_filename: string;
  };
  model_name: string;
  sample_count: number;
  primary_metric: { name: string; value: number };
  secondary_metrics: Record<string, number | null>;
  per_class_f1: Record<string, number>;
};

type HistoryItem = {
  id: number;
  model_name: string;
  submission_type: string;
  source_filename: string;
  sample_count: number;
  macro_f1: number;
  accuracy: number;
  balanced_accuracy: number;
  weighted_f1: number;
  quadratic_kappa: number | null;
  created_at: string;
};

type LeaderboardItem = {
  model_name: string;
  best_macro_f1: number;
  best_quadratic_kappa: number | null;
  best_macro_precision: number | null;
  best_macro_recall: number | null;
  best_accuracy: number;
  total_runs: number;
  last_run_at: string;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

function formatMetric(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "N/A";
  return value.toFixed(4);
}

function formatDate(value: string) {
  return new Date(value).toLocaleString();
}

function App() {
  const [modelName, setModelName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<EvaluationResponse | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [leaderboard, setLeaderboard] = useState<LeaderboardItem[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const evaluatedModelCount = leaderboard.length;
  const evaluatedRunCount = history.length;
  const bestMacroF1 = leaderboard.length > 0 ? Math.max(...leaderboard.map((item) => item.best_macro_f1)) : null;
  const quadraticKappaScores = leaderboard
    .map((item) => item.best_quadratic_kappa)
    .filter((value): value is number => value !== null && value !== undefined);
  const bestQuadraticKappa =
    quadraticKappaScores.length > 0 ? Math.max(...quadraticKappaScores) : null;

  async function loadDashboardData() {
    const [historyResponse, leaderboardResponse] = await Promise.all([
      fetch(`${API_BASE}/api/history`),
      fetch(`${API_BASE}/api/leaderboard`),
    ]);

    if (historyResponse.ok) {
      const historyPayload = await historyResponse.json();
      setHistory(historyPayload.items ?? []);
    }

    if (leaderboardResponse.ok) {
      const leaderboardPayload = await leaderboardResponse.json();
      setLeaderboard(leaderboardPayload.items ?? []);
    }
  }

  useEffect(() => {
    loadDashboardData().catch(() => {
      setError("Could not load saved history from the backend.");
    });
  }, []);

  async function submitEvaluation(event: FormEvent) {
    event.preventDefault();
    if (!modelName.trim() || !file) return;

    setLoading(true);
    setError("");
    setResult(null);

    const form = new FormData();
    form.append("model_name", modelName.trim());
    form.append("predictions", file);

    try {
      const response = await fetch(`${API_BASE}/api/evaluate`, {
        method: "POST",
        body: form,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Evaluation failed");
      setResult(payload);
      await loadDashboardData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Evaluation failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="logo-strip" aria-label="Institution logos">
          <img src="/logos/tanuh_logo.png" alt="TANUH" />
          <img src="/logos/moe_logo.png" alt="Ministry of Education" />
          <img src="/logos/iisc_logo.png" alt="Indian Institute of Science" />
        </div>
      </header>

      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Mammogram benchmark tool</p>
          <h1>Breast Density Benchmark Tool</h1>
          <p>
            Upload CSV files from your SVM, CNN, or PyTorch pipeline. The tool evaluates them
            against the benchmark answer key, saves the run history, and updates the leaderboard.
          </p>
        </div>
        <form className="evaluation-panel" onSubmit={submitEvaluation}>
          <label>
            Model name
            <input
              value={modelName}
              onChange={(event) => setModelName(event.target.value)}
              placeholder="SVM baseline"
            />
          </label>
          <label>
            Prediction CSV
            <input
              type="file"
              accept=".csv,text/csv"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            />
          </label>
          <p className="helper-text">
            Upload the prediction CSV with image_id,prediction or image_id,p0,p1,p2,p3 columns.
          </p>
          <button type="submit" disabled={loading || !file || !modelName.trim()}>
            <Upload size={18} />
            {loading ? "Evaluating" : "Proceed"}
          </button>
          {error && <p className="error">{error}</p>}
        </form>
      </section>

      <section className="workflow">
        <article>
          <Database size={26} />
          <h2>Benchmark set</h2>
          <p>Evaluate predictions across the balanced EMBED and IBIA test set with 200 cases per density class.</p>
        </article>
        <article>
          <History size={26} />
          <h2>Saved history</h2>
          <p>Each evaluation run is stored so you can compare repeated submissions from different model names.</p>
        </article>
        <article>
          <Trophy size={26} />
          <h2>Leaderboard</h2>
          <p>Track best model performance with macro F1 as the main score and accuracy as a visible summary.</p>
        </article>
      </section>

      <section className="overview">
        <article>
          <ClipboardCheck size={24} />
          <div>
            <span>Models evaluated</span>
            <strong>{evaluatedModelCount}</strong>
            <p>{evaluatedRunCount} saved upload{evaluatedRunCount === 1 ? "" : "s"}</p>
          </div>
        </article>
        <article>
          <Images size={24} />
          <div>
            <span>Benchmark images</span>
            <strong>800</strong>
            <p>A/B/C/D balanced at 200 each</p>
          </div>
        </article>
        <article>
          <Gauge size={24} />
          <div>
            <span>Best macro F1</span>
            <strong>{formatMetric(bestMacroF1)}</strong>
            <p>Primary leaderboard metric</p>
          </div>
        </article>
        <article>
          <Award size={24} />
          <div>
            <span>Best kappa</span>
            <strong>{formatMetric(bestQuadraticKappa)}</strong>
            <p>Quadratic weighted agreement</p>
          </div>
        </article>
      </section>

      <section className="density-guide">
        <div className="density-image" tabIndex={0} aria-label="Density reference image. Hover or focus to see details.">
          <img src="/density-labels-corrected.png" alt="Mammogram examples for density labels A, B, C, and D" />
          <div className="density-image-caption">
            <span>Density reference</span>
            <p>A visual guide for understanding the benchmark classes used during evaluation.</p>
          </div>
        </div>
        <div className="density-content">
          <div className="density-heading">
            <p className="eyebrow">Breast Density label guide</p>
          </div>
          <div className="density-class-grid" aria-label="Breast density class descriptions">
            <article>
              <strong>A</strong>
              <div>
                <h3>Almost entirely fatty</h3>
                <p>Low fibroglandular density with mostly fatty tissue appearance.</p>
              </div>
            </article>
            <article>
              <strong>B</strong>
              <div>
                <h3>Scattered density</h3>
                <p>Scattered areas of fibroglandular tissue mixed with fatty regions.</p>
              </div>
            </article>
            <article>
              <strong>C</strong>
              <div>
                <h3>Heterogeneously dense</h3>
                <p>More dense tissue is visible, making classification more challenging.</p>
              </div>
            </article>
            <article>
              <strong>D</strong>
              <div>
                <h3>Extremely dense</h3>
                <p>High fibroglandular density with brighter tissue across the mammogram.</p>
              </div>
            </article>
          </div>
        </div>
      </section>

      {result && (
        <section className="results">
          <div>
            <p className="eyebrow">Evaluation complete</p>
            <h2>{result.model_name}</h2>
            <p>
              {result.sample_count} matched benchmark samples • saved as run #{result.run.id}
            </p>
          </div>
          <div className="metric primary">
            <span>Macro F1</span>
            <strong>{formatMetric(result.primary_metric.value)}</strong>
          </div>
          {Object.entries(result.secondary_metrics).map(([name, value]) => (
            <div className="metric" key={name}>
              <span>{name.replaceAll("_", " ")}</span>
              <strong>{formatMetric(value)}</strong>
            </div>
          ))}
        </section>
      )}

      <section className="dashboard-grid">
        <section className="table-panel">
          <div className="section-heading">
            <p className="eyebrow">Leaderboard</p>
            <h2>Best score per model</h2>
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Macro F1</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>Kappa</th>
                  <th>Accuracy</th>
                  <th>Runs</th>
                </tr>
              </thead>
              <tbody>
                {leaderboard.map((item) => (
                  <tr key={item.model_name}>
                    <td>{item.model_name}</td>
                    <td>{formatMetric(item.best_macro_f1)}</td>
                    <td>{formatMetric(item.best_macro_precision)}</td>
                    <td>{formatMetric(item.best_macro_recall)}</td>
                    <td>{formatMetric(item.best_quadratic_kappa)}</td>
                    <td>{formatMetric(item.best_accuracy)}</td>
                    <td>{item.total_runs}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {leaderboard.length > 4 && (
            <p className="scroll-note">Scroll inside this table to compare more models.</p>
          )}
        </section>

        <section className="table-panel">
          <div className="section-heading">
            <p className="eyebrow">History</p>
            <h2>Saved evaluations</h2>
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Model</th>
                  <th>Accuracy</th>
                  <th>Time</th>
                </tr>
              </thead>
              <tbody>
                {history.map((item) => (
                  <tr key={item.id}>
                    <td>#{item.id}</td>
                    <td>{item.model_name}</td>
                    <td>{formatMetric(item.accuracy)}</td>
                    <td>{formatDate(item.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {history.length > 5 && (
            <p className="scroll-note">Scroll inside this table to see older runs such as #3, #2, and #1.</p>
          )}
        </section>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
