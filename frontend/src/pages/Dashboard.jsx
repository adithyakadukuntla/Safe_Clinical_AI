import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

function Dashboard() {
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const token = localStorage.getItem("token");

  useEffect(() => {
    if (!token) {
      navigate("/login");
      return;
    }
    fetchJobs();
  }, [token, navigate]);

  const fetchJobs = async () => {
    try {
      const response = await fetch(
        "http://localhost:3000/api/admin/retrain-jobs",
        {
          headers: { Authorization: `Bearer ${token}` },
        },
      );
      const data = await response.json();
      setJobs(data);
    } catch (err) {
      console.error(err);
    }
  };

  const triggerRetraining = async () => {
    setLoading(true);
    try {
      // 1. Notify Node.js to create a job
      const nodeResponse = await fetch(
        "http://localhost:3000/api/admin/trigger-retrain",
        {
          method: "POST",
          headers: { Authorization: `Bearer ${token}` },
        },
      );
      const nodeData = await nodeResponse.json();

      if (!nodeData.jobId) throw new Error("Failed to create job");

      // 2. Trigger Python Dry Run
      const pythonResponse = await fetch(
        "http://localhost:5000/retrain/dry-run",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ jobId: nodeData.jobId }),
        },
      );

      // Refresh jobs list to show new status
      fetchJobs();
    } catch (err) {
      alert("Error starting retraining: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleApprove = async () => {
    if (!window.confirm("Are you sure you want to replace the current model?"))
      return;
    try {
      const res = await fetch("http://localhost:5000/retrain/approve", {
        method: "POST",
      });
      if (res.ok) {
        alert("Model approved successfully!");
        fetchJobs();
      } else {
        alert("Approval failed");
      }
    } catch (err) {
      alert("Error: " + err.message);
    }
  };

  const handleReject = async () => {
    if (!window.confirm("Discard candidate model?")) return;
    try {
      const res = await fetch("http://localhost:5000/retrain/reject", {
        method: "POST",
      });
      if (res.ok) {
        alert("Model rejected.");
        fetchJobs();
      }
    } catch (err) {
      alert("Error: " + err.message);
    }
  };

  return (
    <div className="app-container">
      <div className="header">
        <h1>Admin Dashboard</h1>
        <p className="subtitle">Manage Model Retraining & Performance</p>
      </div>

      <div style={{ textAlign: "center", marginBottom: "2rem" }}>
        <button
          className="submit-btn"
          onClick={triggerRetraining}
          disabled={loading}
          style={{ maxWidth: "300px" }}
        >
          {loading
            ? "Training in Progress..."
            : "🚀 Trigger Retraining (Dry Run)"}
        </button>
      </div>

      <div className="jobs-list">
        <h2>Retraining History</h2>
        {jobs.map((job) => (
          <div
            key={job._id}
            className="result-card"
            style={{ textAlign: "left", marginBottom: "1rem" }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <h3>Job ID: {job._id.substring(0, 8)}...</h3>
              <span className={`status-badge ${job.status}`}>
                {job.status.toUpperCase()}
              </span>
            </div>
            <p>
              <strong>Triggered By:</strong> {job.triggeredBy}
            </p>
            <p>
              <strong>Start Time:</strong>{" "}
              {new Date(job.startTime).toLocaleString()}
            </p>

            {job.status === "completed" && job.metrics_new && (
              <div
                className="comparison"
                style={{
                  marginTop: "1rem",
                  padding: "1rem",
                  background: "#0f172a",
                  borderRadius: "10px",
                }}
              >
                <h4>Retraining Analysis & Verification</h4>

                {job.detailed_results ? (
                  <div className="optimization-details">
                    {/* Summary Comparison - Only if new format exists */}
                    {job.detailed_results.Phase1_Unconstrained ? (
                      <div
                        style={{
                          marginBottom: "2rem",
                          background: "#0f172a",
                          padding: "1.5rem",
                          borderRadius: "12px",
                          border: "1px solid #334155",
                        }}
                      >
                        <h4
                          style={{
                            marginTop: 0,
                            borderBottom: "1px solid #334155",
                            paddingBottom: "0.5rem",
                            marginBottom: "1rem",
                          }}
                        >
                          🚀 Executive Summary (Weighted Ensembles)
                        </h4>
                        <div
                          style={{
                            display: "grid",
                            gridTemplateColumns: "1fr 1fr",
                            gap: "2rem",
                          }}
                        >
                          <div>
                            <h6
                              style={{
                                color: "#38bdf8",
                                marginBottom: "0.5rem",
                              }}
                            >
                              Baseline (Data-Driven)
                            </h6>
                            <div
                              style={{
                                fontSize: "1.5rem",
                                fontWeight: "bold",
                                color: "#e2e8f0",
                              }}
                            >
                              {(
                                job.detailed_results.Phase1_Unconstrained
                                  .metrics.Ensemble_Weighted["ROC-AUC"] * 100
                              ).toFixed(2)}
                              %{" "}
                              <span
                                style={{ fontSize: "0.8rem", color: "#94a3b8" }}
                              >
                                AUC
                              </span>
                            </div>
                          </div>
                          <div>
                            <h6
                              style={{
                                color: "#4ade80",
                                marginBottom: "0.5rem",
                              }}
                            >
                              Optimized (Constrained)
                            </h6>
                            <div
                              style={{
                                fontSize: "1.5rem",
                                fontWeight: "bold",
                                color: "#e2e8f0",
                              }}
                            >
                              {(
                                job.detailed_results.Phase2_Constrained.metrics
                                  .Ensemble_Weighted["ROC-AUC"] * 100
                              ).toFixed(2)}
                              %{" "}
                              <span
                                style={{ fontSize: "0.8rem", color: "#94a3b8" }}
                              >
                                AUC
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div
                        style={{
                          padding: "1rem",
                          marginBottom: "1rem",
                          background: "#334155",
                          borderRadius: "8px",
                          color: "#e2e8f0",
                        }}
                      >
                        ⚠️ <strong>Legacy Result Format</strong>. Detailed
                        breakdown not available for this job.
                      </div>
                    )}

                    {/* Detailed Tabs/Sections - Only if new format exists */}
                    {job.detailed_results.Phase1_Unconstrained && (
                      <div
                        style={{
                          display: "grid",
                          gridTemplateColumns: "1fr",
                          gap: "2rem",
                        }}
                      >
                        {/* Phase 1: Unconstrained */}
                        <PhaseSection
                          title="Phase 1: Unconstrained Models (Data-Driven)"
                          color="#38bdf8"
                          description="Models trained purely on data patterns without enforcing medical knowledge constraints."
                          data={job.detailed_results.Phase1_Unconstrained}
                        />

                        {/* Phase 2: Constrained */}
                        <PhaseSection
                          title="Phase 2: Constrained Models (Medical Knowledge)"
                          color="#4ade80"
                          description="Models enforcing monotonic constraints (e.g., Age increases Risk) to ensure safety and interpretability."
                          data={job.detailed_results.Phase2_Constrained}
                        />
                      </div>
                    )}
                  </div>
                ) : (
                  <div
                    style={{
                      padding: "1rem",
                      background: "#1e293b",
                      borderRadius: "8px",
                    }}
                  >
                    <p style={{ color: "#94a3b8" }}>
                      Detailed metrics waiting for next run...
                    </p>
                    <div style={{ marginTop: "0.5rem" }}>
                      <strong>Legacy Metrics:</strong>
                      <span style={{ marginLeft: "1rem" }}>
                        Accuracy: {(job.metrics_new.accuracy * 100).toFixed(2)}%
                      </span>
                      <span style={{ marginLeft: "1rem" }}>
                        ROC-AUC: {(job.metrics_new.roc_auc * 100).toFixed(2)}%
                      </span>
                    </div>
                  </div>
                )}

                <div
                  className="actions"
                  style={{ marginTop: "1.5rem", display: "flex", gap: "1rem" }}
                >
                  <button
                    className="submit-btn"
                    style={{ background: "var(--success)", marginTop: 0 }}
                    onClick={handleApprove}
                  >
                    ✅ Accept & Deploy
                  </button>
                  <button
                    className="submit-btn"
                    style={{ background: "var(--danger)", marginTop: 0 }}
                    onClick={handleReject}
                  >
                    ❌ Reject
                  </button>
                </div>
              </div>
            )}

            {job.logs && job.logs.length > 0 && (
              <div
                style={{
                  marginTop: "1rem",
                  fontSize: "0.85rem",
                  color: "#94a3b8",
                }}
              >
                <strong>Latest Log:</strong> {job.logs[job.logs.length - 1]}
              </div>
            )}
          </div>
        ))}
        {jobs.length === 0 && (
          <p style={{ textAlign: "center", color: "#94a3b8" }}>
            No retraining jobs found.
          </p>
        )}
      </div>
    </div>
  );
}

// Helper Component for Section
const PhaseSection = ({ title, color, description, data }) => {
  // Safety check just in case
  if (!data || !data.metrics || !data.weights) return null;

  const { metrics, weights } = data;

  return (
    <div
      style={{
        background: "#1e293b",
        padding: "1.5rem",
        borderRadius: "12px",
        border: `1px solid ${color}30`,
      }}
    >
      <h5
        style={{
          color: color,
          marginTop: 0,
          marginBottom: "0.5rem",
          fontSize: "1.1rem",
        }}
      >
        {title}
      </h5>
      <p
        style={{
          fontSize: "0.85rem",
          color: "#94a3b8",
          marginBottom: "1.5rem",
        }}
      >
        {description}
      </p>

      {/* Weights */}
      <div
        style={{
          marginBottom: "1.5rem",
          background: "#0f172a",
          padding: "1rem",
          borderRadius: "8px",
          border: "1px solid #334155",
        }}
      >
        <strong
          style={{
            display: "block",
            marginBottom: "0.5rem",
            color: "#cbd5e1",
            fontSize: "0.9rem",
          }}
        >
          Ensemble Weights (Grid Search Optimized)
        </strong>
        <code style={{ color: color, fontSize: "0.9rem" }}>
          XGB: {(weights[0] * 100).toFixed(1)}% | LGB:{" "}
          {(weights[1] * 100).toFixed(1)}% | Cat:{" "}
          {(weights[2] * 100).toFixed(1)}%
        </code>
      </div>

      {/* Table */}
      <div style={{ overflowX: "auto" }}>
        <table
          style={{
            width: "100%",
            textAlign: "left",
            borderCollapse: "collapse",
            color: "#cbd5e1",
            fontSize: "0.85rem",
          }}
        >
          <thead>
            <tr
              style={{
                background: "#0f172a",
                color: "#94a3b8",
                textTransform: "uppercase",
                fontSize: "0.75rem",
              }}
            >
              <th style={{ padding: "0.75rem", borderRadius: "6px 0 0 6px" }}>
                Model Variant
              </th>
              <th style={{ padding: "0.75rem" }}>Accuracy</th>
              <th style={{ padding: "0.75rem" }}>Precision</th>
              <th style={{ padding: "0.75rem" }}>Recall</th>
              <th style={{ padding: "0.75rem" }}>F1-Score</th>
              <th
                style={{
                  padding: "0.75rem",
                  color: "#fff",
                  borderRadius: "0 6px 6px 0",
                }}
              >
                ROC-AUC
              </th>
            </tr>
          </thead>
          <tbody>
            {[
              "XGBoost",
              "LightGBM",
              "CatBoost",
              "Ensemble_Average",
              "Ensemble_Weighted",
            ].map((model) => (
              <tr
                key={model}
                style={{
                  borderBottom: "1px solid #334155",
                  background: model.includes("Weighted")
                    ? `${color}10`
                    : "transparent",
                }}
              >
                <td
                  style={{
                    padding: "0.75rem",
                    fontWeight: model.includes("Ensemble") ? "bold" : "normal",
                    color: model.includes("Weighted") ? color : "inherit",
                  }}
                >
                  {model.replace("_", " ")}
                </td>
                {metrics[model] ? (
                  <>
                    <td style={{ padding: "0.75rem" }}>
                      {(metrics[model].Accuracy * 100).toFixed(2)}%
                    </td>
                    <td style={{ padding: "0.75rem" }}>
                      {(metrics[model].Precision * 100).toFixed(2)}%
                    </td>
                    <td style={{ padding: "0.75rem" }}>
                      {(metrics[model].Recall * 100).toFixed(2)}%
                    </td>
                    <td style={{ padding: "0.75rem" }}>
                      {(metrics[model]["F1-Score"] * 100).toFixed(2)}%
                    </td>
                    <td
                      style={{
                        padding: "0.75rem",
                        fontWeight: "bold",
                        color: model.includes("Weighted") ? color : "inherit",
                      }}
                    >
                      {(metrics[model]["ROC-AUC"] * 100).toFixed(2)}%
                    </td>
                  </>
                ) : (
                  <td
                    colSpan="5"
                    style={{ padding: "0.75rem", color: "#94a3b8" }}
                  >
                    N/A
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default Dashboard;
