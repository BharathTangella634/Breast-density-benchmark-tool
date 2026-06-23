import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Award,
  BarChart3,
  ClipboardCheck,
  Database,
  Download,
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
type SortKey = "best_macro_f1" | "best_accuracy" | "best_macro_precision" | "best_macro_recall" | "best_quadratic_kappa";

const sortOptions: { key: SortKey; label: string }[] = [
  { key: "best_macro_f1", label: "Macro F1" },
  { key: "best_accuracy", label: "Accuracy" },
  { key: "best_macro_precision", label: "Precision" },
  { key: "best_macro_recall", label: "Recall" },
  { key: "best_quadratic_kappa", label: "Kappa" },
];

function formatMetric(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "N/A";
  return value.toFixed(4);
}

function formatDate(value: string) {
  return new Date(value).toLocaleString();
}

function App() {
  const [modelName, setModelName] = useState("");
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [onnxFile, setOnnxFile] = useState<File | null>(null);
  const [result, setResult] = useState<EvaluationResponse | null>(null);
  const [leaderboard, setLeaderboard] = useState<LeaderboardItem[]>([]);
  const [error, setError] = useState("");
  const [onnxMessage, setOnnxMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [onnxLoading, setOnnxLoading] = useState(false);
  const [leaderboardSort, setLeaderboardSort] = useState<SortKey>("best_macro_f1");
  const evaluatedModelCount = leaderboard.length;
  const bestMacroF1 = leaderboard.length > 0 ? Math.max(...leaderboard.map((item) => item.best_macro_f1)) : null;
  const quadraticKappaScores = leaderboard
    .map((item) => item.best_quadratic_kappa)
    .filter((value): value is number => value !== null && value !== undefined);
  const bestQuadraticKappa =
    quadraticKappaScores.length > 0 ? Math.max(...quadraticKappaScores) : null;
  const sortedLeaderboard = useMemo(
    () =>
      [...leaderboard].sort((first, second) => {
        const firstValue = first[leaderboardSort] ?? Number.NEGATIVE_INFINITY;
        const secondValue = second[leaderboardSort] ?? Number.NEGATIVE_INFINITY;
        if (secondValue !== firstValue) return secondValue - firstValue;
        return first.model_name.localeCompare(second.model_name);
      }),
    [leaderboard, leaderboardSort],
  );

  async function loadDashboardData() {
    const leaderboardResponse = await fetch(`${API_BASE}/api/leaderboard`);
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

  async function submitEvaluation() {
    if (!modelName.trim() || !csvFile) return;

    setLoading(true);
    setError("");
    setOnnxMessage("");
    setResult(null);

    const form = new FormData();
    form.append("model_name", modelName.trim());
    form.append("predictions", csvFile);

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

  async function submitOnnxEvaluation() {
    if (!modelName.trim() || !onnxFile) return;

    setOnnxLoading(true);
    setError("");
    setOnnxMessage("");

    const form = new FormData();
    form.append("model_name", modelName.trim());
    form.append("model_file", onnxFile);

    try {
      const response = await fetch(`${API_BASE}/api/evaluate-onnx`, {
        method: "POST",
        body: form,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "ONNX evaluation failed");
      setOnnxMessage(payload.detail || "ONNX model uploaded.");
    } catch (err) {
      setOnnxMessage(err instanceof Error ? err.message : "ONNX evaluation failed");
    } finally {
      setOnnxLoading(false);
    }
  }

  async function submitSelectedEvaluation() {
    if (csvFile && onnxFile) {
      setError("Choose either a prediction CSV or an ONNX model, not both.");
      return;
    }

    if (csvFile) {
      await submitEvaluation();
      return;
    }

    if (onnxFile) {
      await submitOnnxEvaluation();
    }
  }

  function exportLeaderboard() {
    const headers = ["rank", "model", "macro_f1", "precision", "recall", "kappa", "accuracy", "runs"];
    const rows = sortedLeaderboard.map((item, index) => [
      index + 1,
      item.model_name,
      formatMetric(item.best_macro_f1),
      formatMetric(item.best_macro_precision),
      formatMetric(item.best_macro_recall),
      formatMetric(item.best_quadratic_kappa),
      formatMetric(item.best_accuracy),
      item.total_runs,
    ]);
    const csv = [headers, ...rows]
      .map((row) =>
        row
          .map((value) => `"${String(value).replaceAll('"', '""')}"`)
          .join(","),
      )
      .join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "breast_density_leaderboard.csv";
    link.click();
    URL.revokeObjectURL(url);
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
          <h1>
            Breast Density
            <span>Benchmark Tool</span>
          </h1>
          <p>
            Evaluate ready-made prediction CSVs now, or prepare ONNX model submissions for server-side
            inference on the benchmark set.
          </p>
        </div>

        <form
          className="evaluation-panel combined-upload-panel"
          onSubmit={(event) => {
            event.preventDefault();
            submitSelectedEvaluation();
          }}
        >
          <label>
            Model name
            <input
              value={modelName}
              onChange={(event) => setModelName(event.target.value)}
              placeholder="ConvNeXt baseline"
            />
          </label>

          <label className="upload-drop">
            <input
              key={onnxFile ? "csv-reset" : "csv"}
              type="file"
              accept=".csv,text/csv"
              onChange={(event) => {
                const selectedFile = event.target.files?.[0] ?? null;
                setCsvFile(selectedFile);
                if (selectedFile) setOnnxFile(null);
              }}
            />
            <span>Select prediction CSV</span>
            <small>{csvFile ? csvFile.name : "image_id,prediction or probability columns"}</small>
          </label>

          <div className="choice-divider"><span>or</span></div>

          <label className="upload-drop">
            <input
              key={csvFile ? "onnx-reset" : "onnx"}
              type="file"
              accept=".onnx"
              onChange={(event) => {
                const selectedFile = event.target.files?.[0] ?? null;
                setOnnxFile(selectedFile);
                if (selectedFile) setCsvFile(null);
              }}
            />
            <span>Select .onnx model</span>
            <small>{onnxFile ? onnxFile.name : "ONNX file with output order A,B,C,D"}</small>
          </label>

          <button type="submit" disabled={loading || onnxLoading || !modelName.trim() || (!csvFile && !onnxFile)}>
            <Upload size={18} />
            {loading ? "Evaluating" : onnxLoading ? "Checking" : "Evaluate"}
          </button>
          {error && <p className="error">{error}</p>}
          {onnxMessage && <p className="helper-text">{onnxMessage}</p>}
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
          <div className="leaderboard-toolbar" aria-label="Leaderboard controls">
            <div className="sort-controls" aria-label="Sort leaderboard by metric">
              {sortOptions.map((option) => (
                <button
                  className={leaderboardSort === option.key ? "sort-button active" : "sort-button"}
                  key={option.key}
                  onClick={() => setLeaderboardSort(option.key)}
                  type="button"
                >
                  {option.label}
                </button>
              ))}
            </div>
            <button className="export-button" onClick={exportLeaderboard} type="button">
              <Download size={16} />
              Export
            </button>
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Rank</th>
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
                {sortedLeaderboard.map((item, index) => (
                  <tr className={index === 0 ? "top-ranked-row" : ""} key={item.model_name}>
                    <td>#{index + 1}</td>
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
            <p className="scroll-note">Scroll down to see more models.</p>
          )}
        </section>

      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
