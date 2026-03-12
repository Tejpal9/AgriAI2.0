require("dotenv").config();

const express = require("express");
const ejsMate = require("ejs-mate");
const session = require("express-session");
const MongoStore = require("connect-mongo");
const path = require("path");
const methodOverride = require("method-override");
const mongoose = require("mongoose");
const passport = require("passport");
const cookieParser = require("cookie-parser");
const flash = require("connect-flash");
const TimeAgo = require("javascript-time-ago");

const app = express();
require("./config/passport");

const authRoutes = require("./routes/auth.js");
const communityRoute = require("./routes/community.js");
const commentsRouter = require("./routes/comments.js");
const weatherRoute = require("./routes/weatherRoute.js");
const firtilizerRoute = require("./routes/FertilizerRoute.js");

app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "/views"));
app.use(express.urlencoded({ extended: true }));
app.use(methodOverride("_method"));
app.engine("ejs", ejsMate);
app.use(express.static(path.join(__dirname, "/public")));
app.use(express.json());
app.use(cookieParser()); // Add this line
app.use(flash());


const MONGO_URL = process.env.MONGODB_URL;

main()
  .then(() => {
    console.log("connected to DB");
  })
  .catch((err) => {
    console.log(err);
  });

async function main() {
  // console.log("MONGO_URL:", MONGO_URL);
  await mongoose.connect(MONGO_URL);
}

const store = MongoStore.create({
  mongoUrl: MONGO_URL,
  crypto: {
    secret: process.env.SECRET,
  },
  touchAfter: 24 * 3600,
});

store.on("error", () => {
  console.log("ERROR IN MONGO SESSION STORE", err);
});

const sessionOptions = {
  store: store,
  secret: process.env.TOKEN_KEY,
  resave: false,
  saveUninitialized: true,
  cookie: {
    expires: Date.now() + 7 * 24 * 60 * 60 * 1000,
    maxAge: 7 * 24 * 60 * 60 * 1000,
    httpOnly: true,
  },
};

app.use(session(sessionOptions));
app.use(flash());
app.use(passport.initialize());
app.use(passport.session());

// global variables across routes
app.use(async (req, res, next) => {
  try {
    res.locals.login = req.isAuthenticated();
    res.locals.session = req.session;
    res.locals.currentUser = req.user;
    res.locals.success = req.flash("success");
    res.locals.error = req.flash("error");
    next();
  } catch (error) {
    console.log(error);
    res.redirect("/");
  }
});

app.get("/", (req, res) => {
  res.render("home/index.ejs");
});

app.get("/detect", (req, res) => {
  res.render("detection/detect.ejs");
});



const { GoogleGenerativeAI } = require("@google/generative-ai");
const genAI = new GoogleGenerativeAI(process.env.GEMINI_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });
app.get("/gemini", (req, res) => {
  res.render("chatbot/chatbot.ejs");
});

app.post("/gemini", async (req, res) => {

  const si = `
You are an expert in plants and crops.

Your response must be a HTML.

Example :

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>
<body>
    
</body>
</html>
`;

  const prompt = req.body.text;

  const result = await model.generateContent(si + prompt);

  res.send(result.response.text());
});




app.get("/settings", (req, res) => {
  res.render("profile/settings.ejs");
});
app.get("/language", (req, res) => {
  res.render("profile/language.ejs");
});

app.use("/", authRoutes);
app.use("/community", communityRoute);
app.use("/community/:id/comments", commentsRouter);
app.use("/", weatherRoute);
app.use("/fertilizer", firtilizerRoute);


app.listen(8090, () => {
  console.log("server is listening to port 8090");
});

