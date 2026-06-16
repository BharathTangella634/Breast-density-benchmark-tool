import React, { FormEvent, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { Activity, BarChart3, Database, Upload } from "lucide-react";
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
  best_accuracy: number;
  total_runs: number;
  last_run_at: string;
};

const API_BASE = "http://127.0.0.1:8000";
const DEMO_MODE = true;

const DEMO_RESULT: EvaluationResponse = {
  run: {
    id: 12,
    created_at: "2026-06-16T12:00:00Z",
    submission_type: "csv_predictions",
    source_filename: "svm_predictions.csv",
  },
  model_name: "SVM Baseline",
  sample_count: 420,
  primary_metric: { name: "macro_f1", value: 0.8124 },
  secondary_metrics: {
    accuracy: 0.8262,
    balanced_accuracy: 0.8041,
    weighted_f1: 0.8217,
    quadratic_kappa: 0.7843,
  },
  per_class_f1: {
    A: 0.84,
    B: 0.79,
    C: 0.81,
    D: 0.81,
  },
};

const DEMO_HISTORY: HistoryItem[] = [
  {
    id: 12,
    model_name: "SVM Baseline",
    submission_type: "csv_predictions",
    source_filename: "svm_predictions.csv",
    sample_count: 420,
    macro_f1: 0.8124,
    accuracy: 0.8262,
    balanced_accuracy: 0.8041,
    weighted_f1: 0.8217,
    quadratic_kappa: 0.7843,
    created_at: "2026-06-16T12:00:00Z",
  },
  {
    id: 11,
    model_name: "DenseNet121",
    submission_type: "csv_predictions",
    source_filename: "densenet_predictions.csv",
    sample_count: 420,
    macro_f1: 0.8451,
    accuracy: 0.8524,
    balanced_accuracy: 0.8388,
    weighted_f1: 0.8492,
    quadratic_kappa: 0.8065,
    created_at: "2026-06-15T17:40:00Z",
  },
  {
    id: 10,
    model_name: "ResNet50",
    submission_type: "csv_predictions",
    source_filename: "resnet_predictions.csv",
    sample_count: 420,
    macro_f1: 0.7983,
    accuracy: 0.8095,
    balanced_accuracy: 0.7912,
    weighted_f1: 0.8041,
    quadratic_kappa: 0.7611,
    created_at: "2026-06-15T09:20:00Z",
  },
];

const DEMO_LEADERBOARD: LeaderboardItem[] = [
  {
    model_name: "DenseNet121",
    best_macro_f1: 0.8451,
    best_accuracy: 0.8524,
    total_runs: 4,
    last_run_at: "2026-06-15T17:40:00Z",
  },
  {
    model_name: "SVM Baseline",
    best_macro_f1: 0.8124,
    best_accuracy: 0.8262,
    total_runs: 3,
    last_run_at: "2026-06-16T12:00:00Z",
  },
  {
    model_name: "ResNet50",
    best_macro_f1: 0.7983,
    best_accuracy: 0.8095,
    total_runs: 5,
    last_run_at: "2026-06-15T09:20:00Z",
  },
];

function formatMetric(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "N/A";
  return value.toFixed(4);
}

function formatDate(value: string) {
  return new Date(value).toLocaleString();
}

function App() {
  const [modelName, setModelName] = useState(DEMO_RESULT.model_name);
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<EvaluationResponse | null>(DEMO_MODE ? DEMO_RESULT : null);
  const [history, setHistory] = useState<HistoryItem[]>(DEMO_MODE ? DEMO_HISTORY : []);
  const [leaderboard, setLeaderboard] = useState<LeaderboardItem[]>(DEMO_MODE ? DEMO_LEADERBOARD : []);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

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
    if (DEMO_MODE) return;
    loadDashboardData().catch(() => {
      setError("Could not load saved history from the backend.");
    });
  }, []);

  async function submitEvaluation(event: FormEvent) {
    event.preventDefault();
    if (!modelName.trim()) return;

    if (DEMO_MODE) {
      setLoading(true);
      setError("");
      window.setTimeout(() => {
        const demoRunId = history.length + 20;
        const nextResult: EvaluationResponse = {
          ...DEMO_RESULT,
          model_name: modelName.trim(),
          run: {
            ...DEMO_RESULT.run,
            id: demoRunId,
            source_filename: file?.name ?? "demo_predictions.csv",
            created_at: new Date().toISOString(),
          },
        };
        const nextHistoryItem: HistoryItem = {
          id: demoRunId,
          model_name: modelName.trim(),
          submission_type: "csv_predictions",
          source_filename: file?.name ?? "demo_predictions.csv",
          sample_count: nextResult.sample_count,
          macro_f1: nextResult.primary_metric.value,
          accuracy: nextResult.secondary_metrics.accuracy ?? 0,
          balanced_accuracy: nextResult.secondary_metrics.balanced_accuracy ?? 0,
          weighted_f1: nextResult.secondary_metrics.weighted_f1 ?? 0,
          quadratic_kappa: nextResult.secondary_metrics.quadratic_kappa ?? null,
          created_at: nextResult.run.created_at,
        };
        setResult(nextResult);
        setHistory([nextHistoryItem, ...history]);
        setLeaderboard((current) => {
          const existing = current.find((item) => item.model_name === modelName.trim());
          if (!existing) {
            return [
              {
                model_name: modelName.trim(),
                best_macro_f1: nextResult.primary_metric.value,
                best_accuracy: nextResult.secondary_metrics.accuracy ?? 0,
                total_runs: 1,
                last_run_at: nextResult.run.created_at,
              },
              ...current,
            ];
          }

          return current
            .map((item) =>
              item.model_name === modelName.trim()
                ? {
                    ...item,
                    best_macro_f1: Math.max(item.best_macro_f1, nextResult.primary_metric.value),
                    best_accuracy: Math.max(item.best_accuracy, nextResult.secondary_metrics.accuracy ?? 0),
                    total_runs: item.total_runs + 1,
                    last_run_at: nextResult.run.created_at,
                  }
                : item
            )
            .sort((a, b) => b.best_macro_f1 - a.best_macro_f1 || b.best_accuracy - a.best_accuracy);
        });
        setLoading(false);
      }, 500);
      return;
    }

    if (!file) return;

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
          <p className="eyebrow">{DEMO_MODE ? "Demo preview" : "Private mammogram benchmark tool"}</p>
          <h1>Breast Density Benchmark Tool</h1>
          <p>
            Upload prediction CSV files from your SVM, CNN, or PyTorch pipeline. The server evaluates them
            against hidden local labels, saves the run history, and updates the leaderboard.
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
            {DEMO_MODE
              ? "This dummy page uses sample metrics and leaderboard data so you can preview the interface."
              : "First version uses uploaded prediction CSV files. Direct PyTorch model execution can be added later."}
          </p>
          <button type="submit" disabled={loading || (!DEMO_MODE && !file) || !modelName.trim()}>
            <Upload size={18} />
            {loading ? "Evaluating" : DEMO_MODE ? "Show Demo Result" : "Proceed"}
          </button>
          {error && <p className="error">{error}</p>}
        </form>
      </section>

      <section className="workflow">
        <article>
          <Database size={26} />
          <h2>Private subset</h2>
          <p>Keep the EMBED images and true labels local while the website only accepts compact prediction files.</p>
        </article>
        <article>
          <Activity size={26} />
          <h2>Saved history</h2>
          <p>Each evaluation run is stored so you can compare repeated submissions from different model names.</p>
        </article>
        <article>
          <BarChart3 size={26} />
          <h2>Leaderboard</h2>
          <p>Track best model performance with macro F1 as the main score and accuracy as a visible summary.</p>
        </article>
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
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Macro F1</th>
                <th>Accuracy</th>
                <th>Runs</th>
              </tr>
            </thead>
            <tbody>
              {leaderboard.map((item) => (
                <tr key={item.model_name}>
                  <td>{item.model_name}</td>
                  <td>{formatMetric(item.best_macro_f1)}</td>
                  <td>{formatMetric(item.best_accuracy)}</td>
                  <td>{item.total_runs}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section className="table-panel">
          <div className="section-heading">
            <p className="eyebrow">History</p>
            <h2>Saved evaluations</h2>
          </div>
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
        </section>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
