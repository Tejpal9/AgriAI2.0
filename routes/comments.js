const express = require("express");
const router = express.Router({ mergeParams: true });

const { isLoggedIn } = require("../middleware.js");
const Post = require("../models/post.js");
const Comment = require("../models/comment.js");


router.post("/", isLoggedIn, async (req, res) => {
  try {
    
    let post = await Post.findById(req.params.id);
    if (!post) {
      return res.status(404).send("Post not found");
    }

    
    let newComment = new Comment({
      commentText: req.body.commentText, // Creating a new comment Pass as an object
      author: req.user._id,
    });

    await newComment.save();
    post.comments.push(newComment._id);


    await post.save();

    res.redirect(`/community/${post._id}`);
  } catch (error) {
    console.error(error);
    res.status(500).send("Error creating comment");
  }
});


module.exports = router;