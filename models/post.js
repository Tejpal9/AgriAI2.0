const mongoose = require("mongoose");
const Comment = require("./comment");
const Schema = mongoose.Schema;

const postSchema = new Schema({
  title: {
    type: String,
    required: true,
  },
  description: String,

  image: {
    url: String,
    filename: String,
  },
  owner: {
    type: Schema.Types.ObjectId,
    ref: "User",
  },

  comments: [
    //one to many relation
    {
      type: Schema.Types.ObjectId,
      ref: "Comment",
    },
  ],

  likes: { type: Number, default: 0 },
  usersLiked: [{ type: Schema.Types.ObjectId, ref: "User" }],

  createdAt: {
    type: Date,
    required: true,
  },
});

postSchema.post("findOneAndDelete", async (Post) => {
  if (Post) {
    await Comment.deleteMany({ _id: { $in: Post.comments } });
    //je listing ta delete hobe tar reviews array er moddhyer review gulo/the array delete hobe
  }
});

const Post = mongoose.model("Post", postSchema);

module.exports = Post;
