# 🌾 AgriAI – Plant Disease Detection & Farming Assistant

A full‑stack web application that helps farmers detect crop diseases, calculate fertilizers, and access community features using machine learning and AI.  
The backend is powered by Node.js/Express with MongoDB for data storage, and the system integrates a PyTorch model for image‑based plant disease prediction. A Streamlit UI (`cure.py`) provides an interactive disease‑cure interface.

---

## 🌐 Live Demo

🎯 **Visit the app:** [https://plant-disease-web-gfm9.onrender.com/](https://plant-disease-web-gfm9.onrender.com/)

---

## 🚀 Features

- 🧠 **ML‑powered disease detection** – classify plant leaf images with a pre‑trained PyTorch model (`plant-disease-model-complete.pth`)
- 📸 **Image upload and analysis** through the web UI
- 🌱 **Fertilizer calculator** for nutrient recommendations
- 🔍 **Weather alerts** and community posts
- 💬 **Authentication & profiles** using Passport.js
- 📡 **Real‑time streaming** via Streamlit integration (for cure recommendations)
- 🗂️ **MongoDB data storage** with Mongoose models (`UserModel`, `post`, `comment`)
- 🔐 Secure sessions, JWT, and protection against common web vulnerabilities

---

## 🛠️ Technologies Used

| Layer        | Tools / Libraries                        |
|--------------|-------------------------------------------|
| Backend      | Node.js, Express, MongoDB, Mongoose       |
| Auth         | Passport.js, bcryptjs, JWT                |
| Frontend     | EJS templates, CSS, client‑side JS        |
| ML & Python  | PyTorch model, `python-shell`, Streamlit  |
| Storage      | MongoDB (cloud or local), Cloudinary for uploads |
| Utilities    | dotenv, nodemon, concurrently, Axios      |

---

## 📁 Project Structure

```
.
├── app.js             # Express application entry point
├── cure.py            # Streamlit app for cure suggestions
├── models/            # Mongoose schemas
├── routes/            # Express routers (auth, community, weather, etc.)
├── views/             # EJS templates & partials
├── public/            # Static assets (CSS, JS, images)
└── config/            # Passport and other config
```

---

## ⚙️ Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) (≥18)
- [npm](https://www.npmjs.com/)
- [Python 3.10+](https://www.python.org/)
- MongoDB instance (local or Atlas)
- Cloudinary account (optional – for image uploads)

### Installation

1. **Clone the repository**

   ```bash
   git clone <repo-url> AgriAI2.0
   cd AgriAI2.0
   ```

2. **Install Node dependencies**

   ```bash
   npm install
   ```

3. **Set up environment variables**

   Create a `.env` file in the project root:

   ```env
   PORT=3000
   MONGO_URI=mongodb://localhost:27017/your-db
   SESSION_SECRET=someSecret
   CLOUDINARY_CLOUD_NAME=...
   CLOUDINARY_API_KEY=...
   CLOUDINARY_API_SECRET=...
   ```

4. **Install Python packages**

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the app**

   ```bash
   npm start
   ```

   This uses `concurrently` to launch:

   - Express server (`nodemon app.js`)
   - Streamlit app (`streamlit run cure.py …`)

6. **Open in browser**

   - Web UI: http://localhost:3000  
   - Streamlit UI: http://localhost:8501 (headless mode)

---

## 🧪 Testing

Currently there are no automated tests.  
You can add Jest/Mocha for server-side tests or Cypress for end‑to‑end coverage.

---

## 📦 Deployment

- Host Node.js app on Heroku, Vercel, or any cloud provider
- Deploy Streamlit separately (or embed via iframe)
- Ensure environment variables and MongoDB/Cloudinary creds are configured in production

---

## 🤝 Contributing

1. Fork the repo  
2. Create a feature branch (`git checkout -b feature/foo`)  
3. Commit your changes (`git commit -am 'Add foo'`)  
4. Push to the branch (`git push origin feature/foo`)  
5. Open a pull request

---

## 📄 License

This project is licensed under the **ISC License** – see the [LICENSE](LICENSE) file for details.

---

✨ *Happy farming with AgriAI!*  
Feel free to improve the README or add images/screenshots to showcase the app.