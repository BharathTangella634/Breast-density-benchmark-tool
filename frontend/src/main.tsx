import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Award,
  ClipboardCheck,
  Database,
  Download,
  Gauge,
  Globe2,
  History,
  Images,
  Info,
  Linkedin,
  Loader,
  Trophy,
  Twitter,
  Upload,
  Youtube,
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
  best_accuracy: number;
  total_runs: number;
  last_run_at: string;
  submission_type: string | null;
};

type HistoryItem = {
  id: number;
  model_name: string;
  submission_type: string;
  source_filename: string;
};

type QueueInfo = {
  queued: number;
  running: { job_id: string; model_name: string; elapsed_seconds: number } | null;
  avg_inference_seconds: number | null;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";
type SortKey = "best_macro_f1" | "best_accuracy" | "best_quadratic_kappa";
type DensityLabel = "A" | "B" | "C" | "D";

const sortOptions: { key: SortKey; label: string }[] = [
  { key: "best_macro_f1", label: "Macro F1" },
  { key: "best_accuracy", label: "Accuracy" },
  { key: "best_quadratic_kappa", label: "Kappa" },
];

const densityClasses: { label: DensityLabel; title: string; description: string }[] = [
  {
    label: "A",
    title: "Almost entirely fatty",
    description: "Low fibroglandular density with mostly fatty tissue appearance.",
  },
  {
    label: "B",
    title: "Scattered density",
    description: "Scattered areas of fibroglandular tissue mixed with fatty regions.",
  },
  {
    label: "C",
    title: "Heterogeneously dense",
    description: "More dense tissue is visible, making classification more challenging.",
  },
  {
    label: "D",
    title: "Extremely dense",
    description: "High fibroglandular density with brighter tissue across the mammogram.",
  },
];

function formatMetric(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "N/A";
  return value.toFixed(4);
}

function formatDuration(seconds: number) {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return s > 0 ? `${m}m ${s}s` : `${m}m`;
}

function formatSubmissionType(type: string | null | undefined) {
  if (!type) return "—";
  if (type === "csv_predictions") return "CSV";
  if (type.startsWith("onnx")) return "ONNX";
  return type;
}

function scrollToElementSlowly(element: HTMLElement, duration = 1200) {
  const start = window.scrollY;
  const target = element.getBoundingClientRect().top + window.scrollY - 18;
  const distance = target - start;
  const startedAt = performance.now();

  function step(now: number) {
    const elapsed = Math.min((now - startedAt) / duration, 1);
    const eased = 1 - Math.pow(1 - elapsed, 3);
    window.scrollTo(0, start + distance * eased);
    if (elapsed < 1) requestAnimationFrame(step);
  }

  requestAnimationFrame(step);
}

function App() {
  const [modelName, setModelName] = useState("");
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [onnxFile, setOnnxFile] = useState<File | null>(null);
  const [result, setResult] = useState<EvaluationResponse | null>(null);
  const [leaderboard, setLeaderboard] = useState<LeaderboardItem[]>([]);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [error, setError] = useState("");
  const [evaluationStatus, setEvaluationStatus] = useState("");
  const [loading, setLoading] = useState(false);
  const [onnxLoading, setOnnxLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  // const [showRequirements, setShowRequirements] = useState(false);
  const [leaderboardSort, setLeaderboardSort] = useState<SortKey>("best_macro_f1");
  const [activeDensityLabel, setActiveDensityLabel] = useState<DensityLabel | null>(null);
  const [queueInfo, setQueueInfo] = useState<QueueInfo | null>(null);
  const [backendDown, setBackendDown] = useState(false);
  const [successBanner, setSuccessBanner] = useState("");
  const resultRef = useRef<HTMLElement | null>(null);
  // const requirementsRef = useRef<HTMLDivElement | null>(null);

  const evaluatedModelCount = leaderboard.length;
  const csvModelCount = new Set(
    history.filter((item) => item.submission_type === "csv_predictions").map((item) => item.model_name),
  ).size;
  const onnxModelCount = new Set(
    history.filter((item) => item.submission_type.startsWith("onnx_model")).map((item) => item.model_name),
  ).size;
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
    try {
      const [leaderboardResponse, historyResponse] = await Promise.all([
        fetch(`${API_BASE}/api/leaderboard`),
        fetch(`${API_BASE}/api/history`),
      ]);
      if (leaderboardResponse.ok) {
        const leaderboardPayload = await leaderboardResponse.json();
        setLeaderboard(leaderboardPayload.items ?? []);
      }
      if (historyResponse.ok) {
        const historyPayload = await historyResponse.json();
        setHistory(historyPayload.items ?? []);
      }
      setBackendDown(false);
    } catch {
      setBackendDown(true);
    }
  }

  async function loadQueueInfo() {
    try {
      const response = await fetch(`${API_BASE}/api/queue`);
      if (response.ok) setQueueInfo(await response.json());
    } catch {}
  }

  useEffect(() => {
    loadDashboardData();
    loadQueueInfo();
    const dashboardInterval = setInterval(loadDashboardData, 30000);
    const queueInterval = setInterval(loadQueueInfo, 10000);
    return () => {
      clearInterval(dashboardInterval);
      clearInterval(queueInterval);
    };
  }, []);

  useEffect(() => {
    if (result && resultRef.current) scrollToElementSlowly(resultRef.current);
  }, [result]);

  // useEffect(() => {
  //   if (!showRequirements) return;
  //
  //   function closeRequirementsOnOutsideClick(event: PointerEvent) {
  //     if (!requirementsRef.current?.contains(event.target as Node)) {
  //       setShowRequirements(false);
  //     }
  //   }
  //
  //   document.addEventListener("pointerdown", closeRequirementsOnOutsideClick);
  //   return () => {
  //     document.removeEventListener("pointerdown", closeRequirementsOnOutsideClick);
  //   };
  // }, [showRequirements]);

  async function submitEvaluation() {
    if (!modelName.trim() || !csvFile) return;

    setLoading(true);
    setError("");
    setSuccessBanner("");
    setEvaluationStatus("CSV is being evaluated. Please wait.");
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
      setEvaluationStatus("");
      await loadDashboardData();
    } catch (err) {
      setEvaluationStatus("");
      setError(err instanceof Error ? err.message : "Evaluation failed");
    } finally {
      setLoading(false);
    }
  }

  function submitOnnxEvaluation() {
    if (!modelName.trim() || !onnxFile) return;

    setOnnxLoading(true);
    setError("");
    setSuccessBanner("");
    setResult(null);
    setUploadProgress(0);

    const form = new FormData();
    form.append("model_name", modelName.trim());
    form.append("model_file", onnxFile);

    const xhr = new XMLHttpRequest();

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        setUploadProgress(Math.round((event.loaded / event.total) * 100));
      }
    };

    xhr.onload = async () => {
      setUploadProgress(null);
      try {
        const payload = JSON.parse(xhr.responseText);
        if (xhr.status >= 400) {
          setError(payload.detail || "ONNX submission failed");
          setOnnxLoading(false);
          return;
        }
        const waitTime = formatDuration(payload.estimated_wait_seconds ?? 600);
        setSuccessBanner(
          `Your model "${modelName.trim()}" has been submitted successfully. ` +
          `The leaderboard will update in approximately ${waitTime}. ` +
          `You can close this page — results appear automatically.`
        );
        setModelName("");
        setOnnxFile(null);
        setOnnxLoading(false);
        await loadQueueInfo();
      } catch {
        setError("ONNX submission failed — invalid server response.");
        setOnnxLoading(false);
      }
    };

    xhr.onerror = () => {
      setUploadProgress(null);
      setError("Upload failed — could not connect to the server.");
      setOnnxLoading(false);
    };

    xhr.open("POST", `${API_BASE}/api/submit-onnx`);
    xhr.send(form);
  }

  function submitSelectedEvaluation() {
    if (csvFile && onnxFile) {
      setError("Choose either a prediction CSV or an ONNX model, not both.");
      return;
    }
    if (csvFile) {
      if (!csvFile.name.toLowerCase().endsWith(".csv")) {
        setError("Please select a .csv file.");
        return;
      }
      submitEvaluation();
      return;
    }
    if (onnxFile) {
      if (!onnxFile.name.toLowerCase().endsWith(".onnx")) {
        setError("Please select a .onnx file.");
        return;
      }
      submitOnnxEvaluation();
    }
  }

  function exportLeaderboard() {
    const headers = ["rank", "model", "type", "macro_f1", "kappa", "accuracy", "runs"];
    const rows = sortedLeaderboard.map((item, index) => [
      index + 1,
      item.model_name,
      formatSubmissionType(item.submission_type),
      formatMetric(item.best_macro_f1),
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

  const queueLabel = queueInfo
    ? queueInfo.running
      ? `Queue: 1 running${queueInfo.queued > 0 ? `, ${queueInfo.queued} waiting` : ""}`
      : queueInfo.queued > 0
        ? `Queue: ${queueInfo.queued} waiting`
        : "Queue empty"
    : null;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="logo-strip" aria-label="Institution logos">
          <img src="/logos/tanuh_logo.png" alt="TANUH" />
          <img src="/logos/moe_logo.png" alt="Ministry of Education" />
          <img src="/logos/iisc_logo.png" alt="Indian Institute of Science" />
        </div>
      </header>

      {backendDown && (
        <div className="backend-down-banner">
          Cannot connect to the evaluation server. Please check that the backend is running on {API_BASE}.
        </div>
      )}

      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Mammogram benchmark tool</p>
          <h1>
            Breast Density
            <span>Benchmark Tool</span>
          </h1>
          <p>
            Upload prediction CSVs for instant benchmarking, or submit an ONNX model to
            run inference directly on the benchmark dataset.
          </p>

          <div className="guidelines-grid">
            <article className="guideline-card">
              <h3>Prediction CSV</h3>
              <ul>
                <li>Two columns: <strong>image_id</strong> and <strong>predicted_label</strong></li>
                <li>Labels: uppercase A, B, C, or D</li>
                <li>All 400 benchmark images required</li>
                <li>No duplicate image IDs</li>
                <li>Max file size: 25 MB</li>
              </ul>
              <div className="guideline-example">
                <span>Example</span>
                <code>image_id,predicted_label{"\n"}subject_0001,C{"\n"}subject_0002,B</code>
              </div>
            </article>
            <article className="guideline-card">
              <h3>ONNX Model</h3>
              <ul>
                <li>Self-contained <strong>.onnx</strong> file (no .data files)</li>
                <li>Input: float32 grayscale <strong>[1,1,1024,1024]</strong></li>
                <li>Output: 4 scores (A,B,C,D) or class index (0-3)</li>
                <li>Full inference pipeline included</li>
                <li>Max file size: 750 MB</li>
              </ul>
              <div className="guideline-example">
                <span>Note</span>
                <code>Models are queued and evaluated{"\n"}one at a time on the server.{"\n"}Leaderboard updates automatically.</code>
              </div>
            </article>
          </div>
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
                setError("");
                setSuccessBanner("");
              }}
            />
            <span>Select prediction CSV</span>
            <small>{csvFile ? csvFile.name : "CSV with columns: image_id, predicted_label (A/B/C/D)"}</small>
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
                setError("");
                setSuccessBanner("");
              }}
            />
            <span>Select .onnx model</span>
            <small>{onnxFile ? onnxFile.name : "Self-contained ONNX with output order A,B,C,D"}</small>
          </label>

          {queueLabel && (
            <p className="queue-status">{queueLabel}</p>
          )}

          {/* <div className="panel-help" ref={requirementsRef}>
            <button
              type="button"
              className="help-trigger"
              aria-label="Show submission requirements"
              aria-expanded={showRequirements}
              onClick={() => setShowRequirements((isVisible) => !isVisible)}
            >
              <Info size={16} />
              <span>Submission Guidelines</span>
            </button>
            {showRequirements && (
              <div className="help-popover" role="dialog" aria-label="Submission requirements">
                <div className="requirement-list">
                  <p>
                    <span>CSV format</span>
                    Two columns: image_id and predicted_label. Labels must be A, B, C, or D. All 400 benchmark images required. No duplicates.
                  </p>
                  <p>
                    <span>CSV example</span>
                    image_id,predicted_label<br/>
                    subject_0001,C<br/>
                    subject_0002,B
                  </p>
                  <p>
                    <span>ONNX requirements</span>
                    Standalone .onnx file with full inference pipeline (no separate .data files). Input: float32 grayscale tensor [1,1,1024,1024]. Output: 4 scores (A,B,C,D order) or class index (0=A, 1=B, 2=C, 3=D). Max 750 MB.
                  </p>
                  <p>
                    <span>Self-contained models only</span>
                    The .onnx file must embed all weights. If your export creates a separate .data file, re-export with embedded weights (PyTorch: no external_data threshold; ONNX: save_as_external_data=False).
                  </p>
                  <p>
                    <span>Model name</span>
                    Each model name must be unique. You cannot resubmit with the same name.
                  </p>
                  <p>
                    <span>ONNX queue</span>
                    ONNX models are queued and evaluated one at a time. After uploading, you will see the estimated evaluation time. The leaderboard updates automatically when evaluation completes.
                  </p>
                </div>
              </div>
            )}
          </div> */}

          <button type="submit" disabled={loading || onnxLoading || !modelName.trim() || (!csvFile && !onnxFile)}>
            {(loading || onnxLoading) ? <Loader size={18} className="spinner" /> : <Upload size={18} />}
            {loading ? "Evaluating CSV..." : onnxLoading ? "Uploading..." : "Evaluate"}
          </button>

          {uploadProgress !== null && (
            <div className="upload-progress-container">
              <div className="upload-progress-bar" style={{ width: `${uploadProgress}%` }} />
              <span className="upload-progress-label">Uploading: {uploadProgress}%</span>
            </div>
          )}

          {evaluationStatus && <p className="status-text">{evaluationStatus}</p>}
          {successBanner && <p className="success-banner">{successBanner}</p>}
          {error && <p className="error">{error}</p>}
        </form>
      </section>

      <section className="workflow">
        <article>
          <Database size={26} />
          <h2>Benchmark set</h2>
          <p>Evaluate predictions across the Benchmark test dataset.</p>
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
            <p>
              CSV {csvModelCount} • ONNX {onnxModelCount}
            </p>
          </div>
        </article>
        <article>
          <Images size={24} />
          <div>
            <span>Benchmark images</span>
            <strong>200</strong>
            <p>A/B/C/D </p>
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
        <div
          className="density-image"
          tabIndex={0}
          aria-label="Density reference image. Hover or focus on A, B, C, or D to highlight the matching description."
          onMouseLeave={() => setActiveDensityLabel(null)}
          onBlur={(event) => {
            if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
              setActiveDensityLabel(null);
            }
          }}
        >
          <img src="/density-labels-cd-best.png" alt="Mammogram examples for density labels A, B, C, and D" />
          {densityClasses.map((densityClass) => (
            <button
              key={densityClass.label}
              type="button"
              className={`density-hotspot density-hotspot-${densityClass.label.toLowerCase()} ${
                activeDensityLabel === densityClass.label ? "active" : ""
              }`}
              aria-label={`Highlight density ${densityClass.label}: ${densityClass.title}`}
              onMouseEnter={() => setActiveDensityLabel(densityClass.label)}
              onFocus={() => setActiveDensityLabel(densityClass.label)}
            >
              <span>{densityClass.label}</span>
            </button>
          ))}
          <div className="density-image-caption">
            <span>{activeDensityLabel ? `Density ${activeDensityLabel}` : "Density reference"}</span>
            <p>
              {activeDensityLabel
                ? densityClasses.find((densityClass) => densityClass.label === activeDensityLabel)?.title
                : "A visual guide for understanding the benchmark classes used during evaluation."}
            </p>
          </div>
        </div>
        <div className="density-content">
          <div className="density-heading">
            <p className="eyebrow">Breast Density label guide</p>
            <p className="density-view-note">Reference examples use left CC images.</p>
          </div>
          <div className="density-class-grid" aria-label="Breast density class descriptions">
            {densityClasses.map((densityClass) => (
              <article
                key={densityClass.label}
                className={activeDensityLabel === densityClass.label ? "active" : ""}
                onMouseEnter={() => setActiveDensityLabel(densityClass.label)}
                onMouseLeave={() => setActiveDensityLabel(null)}
                onFocus={() => setActiveDensityLabel(densityClass.label)}
                onBlur={() => setActiveDensityLabel(null)}
                tabIndex={0}
              >
                <strong>{densityClass.label}</strong>
                <div>
                  <h3>{densityClass.title}</h3>
                  <p>{densityClass.description}</p>
                </div>
              </article>
            ))}
          </div>
          <aside className="density-citation" aria-label="Density label citation">
            <Info size={17} aria-hidden="true" />
            <p>
              <strong>Density labels A-D follow the ACR BI-RADS Atlas breast composition
              categories.</strong> Citation: American College of Radiology. <cite>ACR
              BI-RADS Atlas: Breast Imaging Reporting and Data System</cite>.
              5th ed. Reston, VA: American College of Radiology; 2013.
              <a
                href="https://www.acr.org/Clinical-Resources/Clinical-Tools-and-Reference/Reporting-and-Data-Systems/BI-RADS"
                target="_blank"
                rel="noreferrer"
              >
                ACR BI-RADS reference
              </a>
            </p>
          </aside>
        </div>
      </section>

      {result && (
        <section className="results" ref={resultRef}>
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
                  <th>Type</th>
                  <th>Macro F1</th>
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
                    <td><span className={`type-badge ${item.submission_type?.startsWith("onnx") ? "type-onnx" : "type-csv"}`}>{formatSubmissionType(item.submission_type)}</span></td>
                    <td>{formatMetric(item.best_macro_f1)}</td>
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

      <footer className="contact-footer">
        <div className="contact-footer-inner">
          <div className="contact-column">
            <h2>Address</h2>
            <p>
              AI Centre of Excellence in Healthcare<br />
              Indian Institute of Science<br />
              Seventh Floor, TCS Smart-X Hub<br />
              Bengaluru, India - 560 012
            </p>
          </div>

          <div className="contact-column contact-column-right">
            <h2>Contact Information</h2>
            <p>
              Study: <a href="mailto:breastcancerdetection@tanuh.ai">breastcancerdetection@tanuh.ai</a>
            </p>
            <p>
              General: <a href="mailto:info@tanuh.ai">info@tanuh.ai</a>
            </p>
            <p>Tel: (080) 2293 4106&nbsp;&nbsp;|&nbsp;&nbsp;(080) 2293 4107</p>
          </div>

          <nav className="social-links" aria-label="TANUH social links">
            <a aria-label="TANUH website" href="https://www.tanuh.ai/" rel="noreferrer" target="_blank">
              <Globe2 size={18} />
            </a>
            <a aria-label="TANUH LinkedIn" href="https://www.linkedin.com/company/tanuh-aicoe/" rel="noreferrer" target="_blank">
              <Linkedin size={18} />
            </a>
            <a aria-label="TANUH X" href="https://x.com/TANUH_AI" rel="noreferrer" target="_blank">
              <Twitter size={18} />
            </a>
            <a aria-label="TANUH YouTube" href="https://www.youtube.com/@TANUH-AI" rel="noreferrer" target="_blank">
              <Youtube size={18} />
            </a>
          </nav>
        </div>
      </footer>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
