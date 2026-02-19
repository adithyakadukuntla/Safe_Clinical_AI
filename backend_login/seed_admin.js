const mongoose = require("mongoose");
const bcrypt = require("bcryptjs");
const User = require("./models/User");
const dotenv = require("dotenv");

dotenv.config();

mongoose
  .connect(process.env.MONGO_URI || "mongodb://127.0.0.1:27017/icu_project")
  .then(() => console.log("MongoDB Connected"))
  .catch((err) => console.log(err));

const seedAdmin = async () => {
  try {
    const existingUser = await User.findOne({ username: "admin" });
    if (existingUser) {
      console.log("Admin user already exists");
      process.exit();
    }

    const hashedPassword = await bcrypt.hash("admin123", 8);
    const admin = new User({
      username: "admin",
      password: hashedPassword,
      role: "admin",
    });

    await admin.save();
    console.log("Admin user created: admin / admin123");
    process.exit();
  } catch (error) {
    console.error("Error seeding admin:", error);
    process.exit(1);
  }
};

seedAdmin();
