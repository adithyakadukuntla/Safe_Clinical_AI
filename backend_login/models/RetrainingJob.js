const mongoose = require("mongoose");

const RetrainingJobSchema = new mongoose.Schema({
  status: {
    type: String,
    enum: [
      "pending",
      "training",
      "completed",
      "failed",
      "approved",
      "rejected",
    ],
    default: "pending",
  },
  triggeredBy: {
    type: String, // username
    required: true,
  },
  startTime: {
    type: Date,
    default: Date.now,
  },
  endTime: {
    type: Date,
  },
  metrics_current: {
    accuracy: Number,
    f1_score: Number,
    roc_auc: Number,
  },
  metrics_new: {
    accuracy: Number,
    f1_score: Number,
    roc_auc: Number,
  },
  detailed_results: {
    type: mongoose.Schema.Types.Mixed,
  },
  candidate_model_path: {
    type: String,
  },
  logs: [String],
});

module.exports = mongoose.model("RetrainingJob", RetrainingJobSchema);
