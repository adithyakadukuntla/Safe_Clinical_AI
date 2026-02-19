const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const dotenv = require("dotenv");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");

const User = require("./models/User");
const RetrainingJob = require("./models/RetrainingJob");

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

// MongoDB Connection
// MongoDB Connection
const mongoURI =
  process.env.MONGO_URI || "mongodb://127.0.0.1:27017/icu_project";
console.log("Connecting to MongoDB at:", mongoURI);

mongoose
  .connect(mongoURI)
  .then(() => console.log("MongoDB Connected"))
  .catch((err) => console.log("MongoDB Connection Error:", err));

const SECRET_KEY = process.env.JWT_SECRET || "supersecretkey123";

// Middleware to verify token
const verifyToken = (req, res, next) => {
  const token = req.headers["authorization"];
  if (!token) return res.status(403).json({ message: "No token provided" });

  jwt.verify(token.split(" ")[1], SECRET_KEY, (err, decoded) => {
    if (err)
      return res.status(500).json({ message: "Failed to authenticate token" });
    req.userId = decoded.id;
    req.username = decoded.username;
    next();
  });
};

// --- Auth Routes ---

app.post("/api/auth/register", async (req, res) => {
  const { username, password } = req.body;
  try {
    const hashedPassword = await bcrypt.hash(password, 8);
    const user = new User({ username, password: hashedPassword });
    await user.save();
    res.status(201).json({ message: "User registered successfully" });
  } catch (error) {
    res.status(500).json({ message: "Error registering user", error });
  }
});

app.post("/api/auth/login", async (req, res) => {
  const { username, password } = req.body;
  try {
    const user = await User.findOne({ username });
    if (!user) return res.status(404).json({ message: "User not found" });

    const passwordIsValid = await bcrypt.compare(password, user.password);
    if (!passwordIsValid)
      return res.status(401).json({ message: "Invalid password" });

    const token = jwt.sign(
      { id: user._id, username: user.username },
      SECRET_KEY,
      {
        expiresIn: 86400, // 24 hours
      },
    );

    res.status(200).json({ auth: true, token, username: user.username });
  } catch (error) {
    res.status(500).json({ message: "Error logging in", error });
  }
});

// --- Retraining Routes ---

// Get all jobs
app.get("/api/admin/retrain-jobs", verifyToken, async (req, res) => {
  try {
    const jobs = await RetrainingJob.find().sort({ startTime: -1 });
    res.status(200).json(jobs);
  } catch (error) {
    res.status(500).json({ message: "Error fetching jobs", error });
  }
});

// Create a new job (Trigger Retraining)
app.post("/api/admin/trigger-retrain", verifyToken, async (req, res) => {
  try {
    const job = new RetrainingJob({
      triggeredBy: req.username,
      status: "pending",
    });
    await job.save();

    // In a real scenario, we might trigger the Python script via child_process here
    // or call the Python API. For this architecture, let's call Python API.

    // We will return the job ID so the frontend can poll or the Python backend can update it
    res.status(200).json({ message: "Retraining triggered", jobId: job._id });
  } catch (error) {
    res.status(500).json({ message: "Error triggering retraining", error });
  }
});

// Update Job Status (Called by Python Backend)
app.post("/api/admin/update-job/:id", async (req, res) => {
  const { id } = req.params;
  const { status, metrics_current, metrics_new, log_message } = req.body;

  try {
    const updateData = {};
    if (status) updateData.status = status;
    if (metrics_current) updateData.metrics_current = metrics_current;
    if (metrics_new) updateData.metrics_new = metrics_new;
    // Handle full results
    if (req.body.full_results)
      updateData.detailed_results = req.body.full_results;

    if (status === "completed" || status === "failed")
      updateData.endTime = Date.now();

    const job = await RetrainingJob.findByIdAndUpdate(
      id,
      {
        $set: updateData,
        $push: log_message ? { logs: log_message } : {},
      },
      { new: true },
    );

    res.status(200).json(job);
  } catch (error) {
    res.status(500).json({ message: "Error updating job", error });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
