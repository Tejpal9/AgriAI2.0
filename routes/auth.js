require("dotenv").config();
const express = require("express");
const router = express.Router();
const User = require("../models/UserModel");
// const LocalStrategy = require("passport-local").Strategy;
const passport = require("passport");
const { saveRedirectUrl } = require("../middleware");

// User registration
router.get("/signup", (req, res) => {
  res.render("user/signup.ejs");
});

router.post(
  "/signup",
  [
    passport.authenticate("local.signup", {
      successRedirect: "/",
      failureRedirect: "/signup",
      failureFlash: true,
    }),
  ],
  async (req, res) => {
    try {
      req.login(newUser, (err) => {
        if (err) {
          return next(err);
        }
        req.flash("success", "Welcome to Website");
        res.redirect("/");
      });
    } catch (e) {
      console.log(e);

      res.redirect("/signup");
    }
  }
);

// User login
router.get("/login", (req, res) => {
  res.render("user/login.ejs");
});

router.post(
  "/login",
  [
    saveRedirectUrl,
    passport.authenticate("local.signin", {
      failureRedirect: "/login",
      failureFlash: true,
    }),
  ],
  async (req, res) => {
    try {
      // redirect to old URL before signing in
      console.log(req.user._id);
      req.flash("success", "Welcome back to Website!");
      let redirectUrl = res.locals.redirectUrl || "/";
      res.redirect(redirectUrl);
    } catch (err) {
      console.log(err);

      return res.redirect("/");
    }
  }
);

router.get("/logout", (req, res, next) => {
  req.logOut((err) => {
    if (err) {
      return next(err);
    }
    req.flash("success", "You are logged out!");
    res.redirect("/");
  });
});

module.exports = router;
