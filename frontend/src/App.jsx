import { useState } from "react";
import "./App.css";

const FEATURES = {
  age: {
    name: "Age",
    info: "Patient age in years",
    unit: "years",
    min: 16,
    max: 89,
    median: 65,
  },
  bmi: {
    name: "BMI",
    info: "Body Mass Index",
    unit: "kg/m²",
    min: 14.84,
    max: 67.81,
    median: 27.65,
  },
  gcs_eyes_apache: {
    name: "GCS Eyes",
    info: "Glasgow Coma Scale Eyes (1-4)",
    unit: "",
    min: 1,
    max: 4,
    median: 4,
  },
  gcs_motor_apache: {
    name: "GCS Motor",
    info: "Glasgow Coma Scale Motor (1-6)",
    unit: "",
    min: 1,
    max: 6,
    median: 6,
  },
  gcs_verbal_apache: {
    name: "GCS Verbal",
    info: "Glasgow Coma Scale Verbal (1-5)",
    unit: "",
    min: 1,
    max: 5,
    median: 5,
  },
  heart_rate_apache: {
    name: "Heart Rate",
    info: "Peak heart rate in first 24h",
    unit: "bpm",
    min: 30,
    max: 178,
    median: 104,
  },
  resprate_apache: {
    name: "Respiratory Rate",
    info: "Peak respiratory rate",
    unit: "/min",
    min: 4,
    max: 60,
    median: 28,
  },
  temp_apache: {
    name: "Temperature",
    info: "Peak temperature",
    unit: "°C",
    min: 32.1,
    max: 39.7,
    median: 36.5,
  },
  map_apache: {
    name: "MAP",
    info: "Lowest Mean Arterial Pressure",
    unit: "mmHg",
    min: 40,
    max: 200,
    median: 67,
  },
  bun_apache: {
    name: "BUN",
    info: "Blood Urea Nitrogen level",
    unit: "mg/dL",
    min: 4,
    max: 127,
    median: 19,
  },
  creatinine_apache: {
    name: "Creatinine",
    info: "Serum creatinine level",
    unit: "mg/dL",
    min: 0.3,
    max: 11.18,
    median: 0.98,
  },
  glucose_apache: {
    name: "Glucose",
    info: "Highest glucose level",
    unit: "mg/dL",
    min: 39,
    max: 598.7,
    median: 133,
  },
  hematocrit_apache: {
    name: "Hematocrit",
    info: "Lowest hematocrit level",
    unit: "%",
    min: 16.2,
    max: 51.4,
    median: 33.2,
  },
  wbc_apache: {
    name: "WBC Count",
    info: "Highest white blood cell count",
    unit: "K/µL",
    min: 0.9,
    max: 45.8,
    median: 10.4,
  },
  sodium_apache: {
    name: "Sodium",
    info: "Highest sodium level",
    unit: "mEq/L",
    min: 117,
    max: 158,
    median: 138,
  },
  potassium_apache: {
    name: "Potassium",
    info: "Highest potassium level",
    unit: "mEq/L",
    min: 3.1,
    max: 7.0,
    median: 4.1,
  },
  bilirubin_apache: {
    name: "Bilirubin",
    info: "Highest bilirubin level",
    unit: "mg/dL",
    min: 0.1,
    max: 51,
    median: 0.6,
  },
  albumin_apache: {
    name: "Albumin",
    info: "Lowest albumin level",
    unit: "g/dL",
    min: 1.2,
    max: 4.6,
    median: 2.9,
  },
};

import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";

function Home() {
  const [formData, setFormData] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setFeedbackSubmitted(false);

    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      alert("Error: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFeedback = async (isCorrect, actualLabel = null) => {
    try {
      const payload = {
        features: formData,
        predicted_label: prediction.status === "Deceased" ? 1 : 0,
        actual_label: isCorrect
          ? prediction.status === "Deceased"
            ? 1
            : 0
          : actualLabel,
      };

      await fetch("http://localhost:5000/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      setFeedbackSubmitted(true);
      alert("Thank you for your feedback!");
    } catch (err) {
      alert("Failed to submit feedback");
    }
  };

  return (
    <div className="app-container">
      <div className="header">
        <h1>🏥 ICU Mortality Risk Predictor</h1>
        <p className="subtitle">
          Real-time AI-powered clinical decision support
        </p>
        <Link
          to="/login"
          style={{
            position: "absolute",
            top: "20px",
            right: "20px",
            color: "var(--text-secondary)",
            textDecoration: "none",
            fontSize: "0.9rem",
          }}
        >
          Admin Login
        </Link>
      </div>

      <form onSubmit={handleSubmit} className="form-grid">
        {Object.entries(FEATURES).map(([key, meta]) => (
          <div key={key} className="input-group">
            <label>
              <span className="label-name">{meta.name}</span>
              <span className="label-info">{meta.info}</span>
            </label>
            <input
              type="number"
              step="0.01"
              min={meta.min}
              max={meta.max}
              placeholder={`${meta.min}-${meta.max} ${meta.unit}`}
              value={formData[key] || ""}
              onChange={(e) =>
                setFormData({ ...formData, [key]: e.target.value })
              }
              required
            />
          </div>
        ))}

        <button type="submit" className="submit-btn" disabled={loading}>
          {loading ? "Analyzing..." : "Predict Risk"}
        </button>
      </form>

      {prediction && (
        <div className={`result-card ${prediction.status.toLowerCase()}`}>
          <h2>🎯 Hybrid Gradient Ensemble Prediction</h2>
          <div className="risk-score">
            <span className="risk-label">Mortality Risk:</span>
            <span className="risk-value">
              {(prediction.mortality_risk * 100).toFixed(2)}%
            </span>
          </div>
          <div className="status-badge">{prediction.status}</div>

          <div className="ensemble-info">
            <h3>📊 Individual Model Predictions</h3>
            <div className="model-grid">
              <div className="model-item">
                <span className="model-name">XGBoost</span>
                <span className="model-prob">
                  {(prediction.confidence.xgboost * 100).toFixed(1)}%
                </span>
              </div>
              <div className="model-item">
                <span className="model-name">LightGBM</span>
                <span className="model-prob">
                  {(prediction.confidence.lightgbm * 100).toFixed(1)}%
                </span>
              </div>
              <div className="model-item">
                <span className="model-name">CatBoost</span>
                <span className="model-prob">
                  {(prediction.confidence.catboost * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            {prediction.ensemble_weights && (
              <div className="fusion-weights">
                <h4>⚖️ Fusion Weights (ROC-AUC based)</h4>
                <div className="weight-bars">
                  <div className="weight-bar">
                    <span>XGBoost</span>
                    <div className="bar-container">
                      <div
                        className="bar-fill"
                        style={{
                          width: `${prediction.ensemble_weights.xgb * 100}%`,
                        }}
                      ></div>
                    </div>
                    <span>
                      {(prediction.ensemble_weights.xgb * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="weight-bar">
                    <span>LightGBM</span>
                    <div className="bar-container">
                      <div
                        className="bar-fill"
                        style={{
                          width: `${prediction.ensemble_weights.lgb * 100}%`,
                        }}
                      ></div>
                    </div>
                    <span>
                      {(prediction.ensemble_weights.lgb * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="weight-bar">
                    <span>CatBoost</span>
                    <div className="bar-container">
                      <div
                        className="bar-fill"
                        style={{
                          width: `${prediction.ensemble_weights.cat * 100}%`,
                        }}
                      ></div>
                    </div>
                    <span>
                      {(prediction.ensemble_weights.cat * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {!feedbackSubmitted && (
            <div
              className="feedback-section"
              style={{
                marginTop: "2rem",
                borderTop: "1px solid var(--border)",
                paddingTop: "1rem",
                textAlign: "center",
              }}
            >
              <h3>Is this prediction correct?</h3>
              <div
                style={{
                  display: "flex",
                  justifyContent: "center",
                  gap: "1rem",
                  marginTop: "1rem",
                }}
              >
                <button
                  className="submit-btn"
                  style={{
                    background: "var(--success)",
                    width: "auto",
                    padding: "0.5rem 2rem",
                    marginTop: 0,
                  }}
                  onClick={() => handleFeedback(true)}
                >
                  Yes, Correct
                </button>
                <button
                  className="submit-btn"
                  style={{
                    background: "var(--danger)",
                    width: "auto",
                    padding: "0.5rem 2rem",
                    marginTop: 0,
                  }}
                  onClick={() => {
                    const actual = prompt(
                      "Please enter actual outcome (1 for Deceased, 0 for Survived):",
                    );
                    if (actual === "0" || actual === "1") {
                      handleFeedback(false, parseInt(actual));
                    }
                  }}
                >
                  No, Incorrect
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {prediction && prediction.shap_html && (
        <div className="shap-container">
          <h2>🔍 AI Explanation (SHAP Force Plot)</h2>
          <p className="shap-desc">
            This plot shows how each feature contributed to the prediction.
            <span className="force-red"> Red</span> bars push the risk higher,
            while
            <span className="force-blue"> Blue</span> bars lower the risk.
          </p>
          <div className="shap-plot-wrapper">
            <iframe
              title="SHAP Force Plot"
              srcDoc={prediction.shap_html}
              className="shap-iframe"
              sandbox="allow-scripts"
            />
          </div>
        </div>
      )}
    </div>
  );
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
    </Router>
  );
}

export default App;
