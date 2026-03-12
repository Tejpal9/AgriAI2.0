const express = require("express");
const router = express.Router();
const {isLoggedIn} = require("../middleware");

const Post = require("../models/post.js");

const multer = require("multer");
const { storage } = require("../cloudConfig.js");
const upload = multer({ storage });




router.get("/", async (req, res) => {
  const allPosts = await Post.find({}).populate("owner").sort({ _id: -1 });
  res.render("community/index.ejs", { allPosts  });
});

router.get("/new",isLoggedIn, (req, res) => {
    
  res.render("community/new.ejs");
});

router.post("/",isLoggedIn, upload.single("post[image]"), async (req, res) => {
  const newPost = new Post(req.body.post);
  // for (let file of req.files) {
  //   let url = file.path;
  //   let filename = file.filename;
  //   newPost.image = { url, filename };
  // }
  if (req.file != undefined) {
    let url = req.file.path;
    let filename = req.file.filename;
    newPost.image = { url, filename };
  }
    newPost.owner = req.user._id;
    //  console.log(req.user);
  newPost.createdAt = new Date();
  let savePost = await newPost.save();
    console.log(savePost);
    req.flash("success", "New Post Created!");
  res.redirect("/community");
});

router.get("/:id", async (req, res) => {
  let { id } = req.params;
  const post = await Post.findById(id)
    .populate("owner") 
    .populate({
      path: "comments",
      populate: {
        path: "author",
      },
    });

  res.render("community/show.ejs", { post });
});

// Like/Dislike Post
router.post("/like/:id", isLoggedIn, async (req, res) => {
  const userId = req.body.userId; // In a real scenario, you'd get this from a logged-in user session.
  const post = await Post.findById(req.params.id);

  if (!post.usersLiked.includes(userId)) {
    post.likes += 1;
    post.usersLiked.push(userId);
  } else {
    post.likes -= 1;
    post.usersLiked = post.usersLiked.filter((id) => id != userId);
  }

  await post.save();
  res.json({ likes: post.likes });
});


router.delete("/:id",isLoggedIn, async (req, res) => {
  let { id } = req.params;
  let deletedPost = await Post.findByIdAndDelete(id);
  console.log(deletedPost);
  res.redirect("/community");
});

module.exports = router;