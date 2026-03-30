import React, { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";

function App() {
  useEffect(() => {
  const audio = document.getElementById("bg-music");

  const playMusic = () => {
    audio.volume = 0.4; // 🔊 louder sound (40%)
    audio.play().catch(() => {});
    document.removeEventListener("click", playMusic);
  };

  document.addEventListener("click", playMusic);

  return () => {
    document.removeEventListener("click", playMusic);
  };
}, []);

  const [musicOn, setMusicOn] = useState(false);
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  const handlePredict = async () => {
    if (!image) return alert("Please upload a leaf image first!");

    setLoading(true);
    setResult("");

    const formData = new FormData();
    formData.append("file", image);

    try {
      const res = await axios.post(
        "https://guava-sense-ml.onrender.com/predict",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      setResult(
        `${res.data.prediction.class} (${(
          res.data.prediction.confidence * 100
        ).toFixed(2)}% confidence)`
      );
    } catch (err) {
      console.error(err);
      setResult("Error: Unable to get prediction");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <audio
  id="bg-music"
  src="/ambient.mp3"
  loop
/>

      <video className="bg-video" autoPlay muted loop>
  <source src="/nature3.mp4" type="video/mp4" />
</video>




      <nav className="navbar">
        <div className="logo" style={{ display: "flex", gap: "15px", alignItems: "center" }}>
  <span><i className="fa fa-leaf"></i> GuavaSense</span>

  <button
    onClick={() => {
      const audio = document.getElementById("bg-music");
      if (!musicOn) {
        audio.volume = 0.15;
        audio.play();
      } else {
        audio.pause();
      }
      setMusicOn(!musicOn);
    }}
    style={{
      background: "rgba(255,255,255,0.15)",
      border: "none",
      borderRadius: "20px",
      padding: "6px 12px",
      color: "white",
      cursor: "pointer",
      fontSize: "0.8rem"
    }}
  >
    {musicOn ? "🔊 Music On" : "🔈 Music Off"}
  </button>
</div>

      </nav>

      <main className="main-section">
        <div className="text-section">
          <h1>Plants make a positive impact on your environment.</h1>
          <p>
            Upload your guava plant leaf to identify diseases instantly with our
            Enhanced detection system.
          </p>

          <div className="buttons">
            <label htmlFor="file-upload" className="upload-btn">
              Upload Image
            </label>
            <input
              id="file-upload"
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              hidden
            />
            <button className="predict-btn" onClick={handlePredict}>
              {loading ? "Predicting..." : "Predict"}
            </button>
          </div>

          {loading && <div className="spinner"></div>}

{result && (
  <div className="result">
    <strong>Prediction Result:</strong> {result}
  </div>
)}

        </div>

        <div className="image-section">
  {preview && (
    <img src={preview} alt="Preview" className="leaf-preview" />
  )}
</div>

      </main>

      {/* About the Project Section */}
      <section className="about-section">
        <h2>About the Project</h2>
        <p>
          Hi this is Sujithkumar S (22BCT0040) and V Gokulakrishnan (22BCE3752){" "}
          <br />
          Together, we built this Guava Plant Disease Detection system to help
          farmers and plant enthusiasts quickly identify diseases in guava
          leaves. Simply upload a guava leaf image, and our AI-powered system
          will detect whether it is Healthy or affected by diseases such as
          Canker, Leaf Spot, Mummification, or Rust, and show the confidence
          percentage for the prediction. Our goal is to provide a simple, fast,
          and reliable tool that can help monitor plant health and support timely
          intervention to protect guava crops.
        </p>
      </section>

      
    </div>
  );
}

export default App;
